#!/usr/bin/env bash
# ==============================================================================
# Script: run_model_screening.sh
# Mục đích: Wrapper kích hoạt thực nghiệm Sàng lọc Mô hình (Checkpoint 2)
# Ghi chú:
# - Gọi entrypoint CLI scripts/run_pipeline.py với cờ --mode screening
# - Cho phép truyền thêm tham số (ví dụ: --device cuda, --gpu-id 0, --mock)
# ==============================================================================

set -euo pipefail

# Kích hoạt .venv nếu chưa được kích hoạt
if [ -z "${VIRTUAL_ENV:-}" ] && [ -f ".venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

# Thiết lập HF_HOME mặc định nếu chưa có
export HF_HOME="${HF_HOME:-$(pwd)/.cache/huggingface}"

echo -e "\n>>> Khởi chạy Sàng lọc Mô hình STT (Model Screening)..."
python scripts/run_pipeline.py --mode screening "$@"
