"""RAGAS offline evaluation (roadmap step 7, README §8/§11).

Runs the retrieval+generation pipeline over the golden set and scores:

  * **context_precision / context_recall** — RAGAS non-LLM variants against
    the golden reference contexts (deterministic string-overlap based; no
    evaluator LLM in the loop). Tracked separately from faithfulness on
    purpose: good retrieval + ungrounded generation is a real failure mode a
    combined metric would hide.
  * **faithfulness_nli** — grounded fraction of answer sentences, computed
    with the same local NLI entailment used by the live grounding guardrail.
    Deterministic and zero-cost.
  * **faithfulness_llm** (optional, --llm-metrics) — RAGAS's LLM-judged
    faithfulness via the local Ollama model. Noisy with a 3B judge; off by
    default.
  * **trap_refusal** — fraction of unanswerable questions correctly refused.

Ablations for the README §11 comparison table (pre/post GraphRAG, pre/post
fine-tuning):

    python eval/ragas_eval.py --tag ft+hybrid
    python eval/ragas_eval.py --tag ft+vector-only --no-graph
    python eval/ragas_eval.py --tag base+hybrid --model llama3.2:3b
    python eval/ragas_eval.py --compare          # markdown table of all runs
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

RESULTS_DIR = Path(__file__).parent / "results"

logger = logging.getLogger("ragas_eval")


def load_golden(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def is_refusal(answer: str, citations: list[str]) -> bool:
    markers = ("does not contain", "cannot answer", "no information",
               "not specify", "does not specify", "not able to answer",
               "manual review", "not mentioned")
    return (not citations) and any(m in answer.lower() for m in markers)


def run_pipeline(golden: list[dict], args) -> list[dict]:
    """Retrieve + generate for every golden question; returns raw records."""
    from agent.llm_client import OllamaClient
    from agent.nodes.generate import SYSTEM_PROMPT, build_user_message, parse_generation
    from retrieval.retriever import HybridRetriever

    retriever = HybridRetriever()
    llm = OllamaClient(model=args.model) if args.model else OllamaClient()
    records = []
    try:
        for i, sample in enumerate(golden, 1):
            question = sample["question"]
            results = retriever.retrieve(
                question,
                top_k=args.top_k,
                use_graph=not args.no_graph,
                use_reranker=not args.no_rerank,
            )
            chunks = [dataclasses.asdict(r) for r in results]
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_message(question, chunks)},
            ]
            answer, citations = parse_generation(llm.chat(messages, json_mode=True))
            records.append({
                "question": question,
                "answer": answer,
                "citations": citations,
                "retrieved_contexts": [c["text"] for c in chunks],
                "cited_contexts": [c["text"] for c in chunks
                                   if c["chunk_id"] in set(citations)],
                "reference": sample["reference"],
                "reference_contexts": sample["reference_contexts"],
                "answerable": sample["answerable"],
            })
            logger.info("[%d/%d] %s -> %d chars, %d citations",
                        i, len(golden), question[:50], len(answer), len(citations))
    finally:
        retriever.close()
    return records


def score_context_metrics(records: list[dict], use_llm: bool, args) -> dict:
    """RAGAS metrics over the answerable samples."""
    from ragas import EvaluationDataset, evaluate
    from ragas.dataset_schema import SingleTurnSample
    from ragas.metrics import NonLLMContextPrecisionWithReference, NonLLMContextRecall

    answerable = [r for r in records if r["answerable"]]
    samples = [
        SingleTurnSample(
            user_input=r["question"],
            response=r["answer"],
            retrieved_contexts=r["retrieved_contexts"],
            reference=r["reference"],
            reference_contexts=r["reference_contexts"],
        )
        for r in answerable
    ]
    metrics = [NonLLMContextPrecisionWithReference(), NonLLMContextRecall()]

    llm_wrapper = None
    if use_llm:
        from langchain_ollama import ChatOllama
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import Faithfulness

        chat = ChatOllama(model=args.model or "medclaim-llm",
                          base_url="http://localhost:11434", temperature=0.0)
        llm_wrapper = LangchainLLMWrapper(chat)
        metrics.append(Faithfulness(llm=llm_wrapper))

    result = evaluate(dataset=EvaluationDataset(samples=samples), metrics=metrics,
                      show_progress=False)
    df = result.to_pandas()
    out = {}
    for col in df.columns:
        if col in ("non_llm_context_precision_with_reference", "non_llm_context_recall",
                   "faithfulness"):
            out[col] = round(float(df[col].mean()), 3)
    return out


def score_nli_faithfulness(records: list[dict]) -> float:
    """Grounded-sentence fraction via the live guardrail's NLI machinery."""
    from agent.nodes.grounding import _get_nli, _sentence_grounded_prob, split_sentences
    from config import get_settings

    if _get_nli() is None:
        return float("nan")
    threshold = get_settings().entailment_threshold
    scores = []
    for r in records:
        if not r["answerable"]:
            continue
        premises = (r["cited_contexts"] or r["retrieved_contexts"])[:6]
        premises = [p[:3000] for p in premises]
        sentences = split_sentences(r["answer"])
        if not premises or not sentences:
            scores.append(0.0)
            continue
        grounded = sum(
            1 for s in sentences
            if max(_sentence_grounded_prob(s, p) for p in premises) >= threshold
        )
        scores.append(grounded / len(sentences))
    return round(sum(scores) / len(scores), 3) if scores else float("nan")


def compare() -> None:
    rows = []
    for path in sorted(RESULTS_DIR.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        rows.append((data["tag"], data["aggregate"]))
    if not rows:
        print("No results in eval/results/ yet.")
        return
    keys = sorted({k for _, agg in rows for k in agg})
    print("| run | " + " | ".join(keys) + " |")
    print("|---|" + "---|" * len(keys))
    for tag, agg in rows:
        print(f"| {tag} | " + " | ".join(str(agg.get(k, "—")) for k in keys) + " |")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--golden", default=str(Path(__file__).parent / "golden.jsonl"))
    parser.add_argument("--tag", default=None, help="name for this run (results file)")
    parser.add_argument("--model", default=None, help="Ollama model override")
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--no-graph", action="store_true")
    parser.add_argument("--no-rerank", action="store_true")
    parser.add_argument("--llm-metrics", action="store_true",
                        help="add RAGAS LLM-judged faithfulness (slow, noisy on 3B)")
    parser.add_argument("--compare", action="store_true",
                        help="print a comparison table of all saved runs")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    if args.compare:
        compare()
        return
    if not args.tag:
        raise SystemExit("--tag is required for an eval run (or use --compare)")

    golden = load_golden(Path(args.golden))
    records = run_pipeline(golden, args)

    aggregate = score_context_metrics(records, args.llm_metrics, args)
    aggregate["faithfulness_nli"] = score_nli_faithfulness(records)
    traps = [r for r in records if not r["answerable"]]
    if traps:
        aggregate["trap_refusal"] = round(
            sum(is_refusal(r["answer"], r["citations"]) for r in traps) / len(traps), 3
        )

    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / f"{args.tag}.json"
    out_path.write_text(json.dumps({
        "tag": args.tag,
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
        "config": {"model": args.model, "top_k": args.top_k,
                   "graph": not args.no_graph, "rerank": not args.no_rerank,
                   "n_samples": len(records)},
        "aggregate": aggregate,
        "samples": records,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n=== {args.tag} ===")
    for k, v in aggregate.items():
        print(f"{k:45s} {v}")
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
