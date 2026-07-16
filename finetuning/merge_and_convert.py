"""Merge the LoRA adapter into the base model and convert to GGUF for Ollama.

Pipeline (run in Colab after qlora_train.py; ~15 min on the free tier):

    adapter + base (fp16) ──merge──> merged HF model
        ──llama.cpp convert_hf_to_gguf──> f16 GGUF
        ──llama-quantize──> q4_K_M GGUF (~2 GB)
        ──push──> your Hugging Face Hub repo

    python finetuning/merge_and_convert.py \
        --base-model meta-llama/Llama-3.2-3B-Instruct \
        --adapter outputs/medclaim-lora \
        --hf-repo <user>/medclaim-llama3.2-3b-gguf

Local deployment afterwards (laptop):

    huggingface-cli download <user>/medclaim-llama3.2-3b-gguf \
        medclaim-llama3.2-3b-q4_K_M.gguf --local-dir finetuning/models
    ollama create medclaim-llm -f finetuning/Modelfile
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

LLAMA_CPP_DIR = Path("llama.cpp")


def run(cmd: list[str]) -> None:
    print("+", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True)


def merge(base_model: str, adapter: str, merged_dir: Path) -> None:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading base {base_model} (fp16) and adapter {adapter} …")
    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.float16, device_map="cpu",
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(model, adapter)
    model = model.merge_and_unload()
    model.save_pretrained(merged_dir, safe_serialization=True)
    AutoTokenizer.from_pretrained(adapter).save_pretrained(merged_dir)
    print(f"Merged model saved to {merged_dir}")


def ensure_llama_cpp() -> None:
    if not LLAMA_CPP_DIR.exists():
        run(["git", "clone", "--depth", "1",
             "https://github.com/ggml-org/llama.cpp", str(LLAMA_CPP_DIR)])
    run([sys.executable, "-m", "pip", "install", "-q", "gguf", "sentencepiece"])
    quantize_bin = LLAMA_CPP_DIR / "build" / "bin" / "llama-quantize"
    if not quantize_bin.exists():
        run(["cmake", "-B", str(LLAMA_CPP_DIR / "build"), "-S", str(LLAMA_CPP_DIR),
             "-DLLAMA_CURL=OFF"])
        run(["cmake", "--build", str(LLAMA_CPP_DIR / "build"),
             "--target", "llama-quantize", "-j", "4"])


def convert_and_quantize(merged_dir: Path, out_dir: Path, quant: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    f16_path = out_dir / "medclaim-llama3.2-3b-f16.gguf"
    quant_path = out_dir / f"medclaim-llama3.2-3b-{quant}.gguf"

    run([sys.executable, str(LLAMA_CPP_DIR / "convert_hf_to_gguf.py"),
         str(merged_dir), "--outfile", str(f16_path), "--outtype", "f16"])
    run([str(LLAMA_CPP_DIR / "build" / "bin" / "llama-quantize"),
         str(f16_path), str(quant_path), quant])
    f16_path.unlink()  # keep only the deployable quant (saves ~6 GB disk)
    print(f"Quantized GGUF: {quant_path}")
    return quant_path


def push_to_hub(gguf_path: Path, adapter_dir: Path, repo_id: str) -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id, repo_type="model", exist_ok=True, private=False)
    api.upload_file(path_or_fileobj=str(gguf_path),
                    path_in_repo=gguf_path.name, repo_id=repo_id)
    metrics = adapter_dir / "training_metrics.json"
    if metrics.exists():
        api.upload_file(path_or_fileobj=str(metrics),
                        path_in_repo="training_metrics.json", repo_id=repo_id)
    print(f"Uploaded to https://huggingface.co/{repo_id}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default="meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument("--adapter", default="outputs/medclaim-lora")
    parser.add_argument("--merged-dir", default="outputs/medclaim-merged")
    parser.add_argument("--out-dir", default="outputs/gguf")
    parser.add_argument("--quant", default="q4_K_M")
    parser.add_argument("--hf-repo", default=None,
                        help="HF repo id to upload the GGUF to (recommended)")
    args = parser.parse_args()

    merged_dir = Path(args.merged_dir)
    if not (merged_dir / "config.json").exists():
        merge(args.base_model, args.adapter, merged_dir)
    ensure_llama_cpp()
    gguf_path = convert_and_quantize(merged_dir, Path(args.out_dir), args.quant)
    if args.hf_repo:
        push_to_hub(gguf_path, Path(args.adapter), args.hf_repo)


if __name__ == "__main__":
    main()
