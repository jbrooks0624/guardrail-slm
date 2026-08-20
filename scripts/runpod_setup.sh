#!/usr/bin/env bash
# Bootstrap this repo on a RunPod GPU pod (L4 or A10G).
set -euo pipefail

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="${HOME}/.local/bin:${PATH}"
fi

cd "$(dirname "$0")/.."
uv python install 3.12
uv sync --dev

echo "guardrail-slm ready on $(uv python find)."
echo "Copy .env.example to .env and set HF_TOKEN / WANDB_API_KEY before training."
echo "CLI: uv run python -m guardrail_slm --help"
echo "GPU extras (torch, bitsandbytes, vllm) are not in the default extra yet."
