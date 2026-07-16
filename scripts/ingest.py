"""CLI for ingesting documents into the MedClaim pipeline.

From the repo root:

    # queue through Celery (a worker must be running):
    python scripts/ingest.py sample_docs/sample_policy.pdf --source-type policy

    # or run the pipeline inline in this process (no worker needed):
    python scripts/ingest.py sample_docs/sample_policy.pdf --sync
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Allow running as `python scripts/ingest.py` from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SUPPORTED = {".pdf", ".docx", ".pptx"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="Document file or directory of documents")
    parser.add_argument(
        "--source-type",
        default="policy",
        choices=["policy", "clinical_guideline", "claim_note"],
    )
    parser.add_argument("--effective-date", default=None, help="ISO date, e.g. 2026-01-01")
    parser.add_argument(
        "--logical-name",
        default=None,
        help="Override the logical document identity (doc_id derivation). "
        "Use when a new version arrives under a different filename.",
    )
    parser.add_argument(
        "--sync",
        action="store_true",
        help="Run the pipeline in-process instead of queueing to Celery",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    target = Path(args.path)
    files = (
        sorted(p for p in target.iterdir() if p.suffix.lower() in SUPPORTED)
        if target.is_dir()
        else [target]
    )
    if not files:
        raise SystemExit(f"No ingestible files found at {target}")

    if args.sync:
        from ingestion.tasks import run_ingestion

        for path in files:
            result = run_ingestion(
                str(path), args.source_type, args.effective_date, args.logical_name
            )
            print(json.dumps(result, indent=2))
    else:
        # Enqueue by task name over a bare Celery client — deliberately does
        # NOT import ingestion.tasks, which would drag in docling/torch just
        # to push a message onto Redis. Only the worker needs the heavy stack.
        from celery import Celery

        from config import get_settings

        settings = get_settings()
        client = Celery(
            broker=settings.celery_broker_url, backend=settings.celery_result_backend
        )
        for path in files:
            async_result = client.send_task(
                "ingestion.ingest_document",
                args=[str(path), args.source_type, args.effective_date, args.logical_name],
            )
            print(f"Queued {path.name} -> task id {async_result.id}")


if __name__ == "__main__":
    main()
