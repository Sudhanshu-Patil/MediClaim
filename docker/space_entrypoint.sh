#!/bin/bash
# Cold-start sequence for the HF Space (Dockerfile.space).
#
# Runs fresh every time the Space wakes from sleep — free Spaces have
# ephemeral storage, so both the Ollama model and any HF-cached embedding/
# NLI/reranker weights are gone after a sleep cycle and re-download here.
# Expect the FIRST request after a wake to take 1-2 minutes; this is the
# accepted tradeoff for $0 hosting (README §10).
set -euo pipefail

echo "[space] starting Ollama…"
ollama serve &
OLLAMA_PID=$!

for i in $(seq 1 30); do
    curl -sf http://127.0.0.1:11434/api/tags >/dev/null 2>&1 && break
    sleep 1
done

echo "[space] pulling fine-tuned GGUF from Hugging Face Hub…"
mkdir -p /app/finetuning/models
huggingface-cli download "${HF_MODEL_REPO:-sud000/medclaim-llama3.2-3b-gguf}" \
    "${HF_MODEL_FILE:-medclaim-llama3.2-3b-q4_K_M.gguf}" \
    --local-dir /app/finetuning/models

echo "[space] registering the model with Ollama…"
ollama create medclaim-llm -f /app/finetuning/Modelfile.cpu

echo "[space] starting the API on port 7860…"
exec uvicorn api.main:app --host 0.0.0.0 --port 7860
