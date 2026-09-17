#!/usr/bin/env bash
# ==============================================================================
# Script: run_streaming_benchmark.sh
# Mục đích: Wrapper kích hoạt thực nghiệm Mô phỏng Trực tuyến Streaming (Checkpoint 4)
# Ghi chú:
# - Gọi entrypoint CLI scripts/run_pipeline.py với cờ --mode streaming
# - Đánh giá LocalAgreement, TTFP, P50/P95 processing latency và Revision Count
# ==============================================================================

set -euo pipefail

# Kích hoạt .venv nếu chưa được kích hoạt
if [ -z "${VIRTUAL_ENV:-}" ] && [ -f ".venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

# Thiết lập HF_HOME mặc định nếu chưa có
export HF_HOME="${HF_HOME:-$(pwd)/.cache/huggingface}"

echo -e "\n>>> Khởi chạy Mô phỏng Trực tuyến Streaming (Streaming Benchmark)..."
python scripts/run_pipeline.py --mode streaming "$@"
