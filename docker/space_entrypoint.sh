#!/bin/bash
# Startup sequence for Dockerfile.space, on a persistent host (Oracle Cloud
# Always Free VM — README §15). Unlike an ephemeral HF Space, storage here
# survives container restarts IF /app/data and /home/appuser/.ollama are
# bind-mounted or named volumes (recommended `docker run -v` flags in
# RUNME) — so the GGUF and HF model cache download once, ever, not on
# every restart. Guards below make that safe either way: idempotent whether
# storage persisted or not.
set -euo pipefail

echo "[space] starting Ollama…"
ollama serve &

for i in $(seq 1 30); do
    curl -sf http://127.0.0.1:11434/api/tags >/dev/null 2>&1 && break
    sleep 1
done

MODEL_FILE="${HF_MODEL_FILE:-medclaim-llama3.2-3b-q4_K_M.gguf}"
mkdir -p /app/finetuning/models
if [ -f "/app/finetuning/models/${MODEL_FILE}" ]; then
    echo "[space] GGUF already present locally, skipping download"
else
    echo "[space] pulling fine-tuned GGUF from Hugging Face Hub (first boot only)…"
    huggingface-cli download "${HF_MODEL_REPO:-sud000/medclaim-llama3.2-3b-gguf}" \
        "$MODEL_FILE" --local-dir /app/finetuning/models
fi

if ollama list 2>/dev/null | grep -q '^medclaim-llm'; then
    echo "[space] medclaim-llm already registered with Ollama"
else
    echo "[space] registering the model with Ollama…"
    ollama create medclaim-llm -f /app/finetuning/Modelfile.cpu
fi

PORT="${PORT:-7860}"
echo "[space] starting the API on port ${PORT}…"
exec uvicorn api.main:app --host 0.0.0.0 --port "$PORT"
