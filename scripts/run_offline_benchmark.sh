#!/usr/bin/env bash
# ==============================================================================
# Script: run_offline_benchmark.sh
# Mục đích: Wrapper kích hoạt thực nghiệm Pipeline Ngoại Tuyến (Checkpoint 3)
# Ghi chú:
# - Gọi entrypoint CLI scripts/run_pipeline.py với cờ --mode offline
# - Đánh giá Baseline Transformers vs faster-whisper (FP16/INT8) và VAD chunking
# ==============================================================================

set -euo pipefail

# Kích hoạt .venv nếu chưa được kích hoạt
if [ -z "${VIRTUAL_ENV:-}" ] && [ -f ".venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

# Thiết lập HF_HOME mặc định nếu chưa có
export HF_HOME="${HF_HOME:-$(pwd)/.cache/huggingface}"

echo -e "\n>>> Khởi chạy Đo kiểm Pipeline Ngoại Tuyến (Offline Benchmark)..."
python scripts/run_pipeline.py --mode offline "$@"
