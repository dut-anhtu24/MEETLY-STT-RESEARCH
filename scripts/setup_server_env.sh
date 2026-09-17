#!/usr/bin/env bash
# ==============================================================================
# Script: setup_server_env.sh
# Mục đích: Thiết lập môi trường ảo Python độc lập (.venv) trên GPU Server
# Ghi chú:
# - Cài đặt package nội bộ trong .venv, TUYỆT ĐỐI KHÔNG cài đặt global hệ thống
# - KHÔNG sửa đổi CUDA system-wide, KHÔNG upgrade driver card màn hình
# - Tự động phát hiện phiên bản CUDA driver để hướng dẫn cài đặt PyTorch GPU phù hợp
# - Cấu hình thư mục cache Hugging Face (HF_HOME) an toàn
# ==============================================================================

set -euo pipefail

# Màu hiển thị
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "\n${CYAN}${BOLD}==============================================================================${NC}"
echo -e "${CYAN}${BOLD}    DUT AI GPU SERVER - THIẾT LẬP MÔI TRƯỜNG THỰC NGHIỆM (MEETLY STT)        ${NC}"
echo -e "${CYAN}${BOLD}==============================================================================${NC}\n"

# 1. Kiểm tra Python 3
if ! command -v python3 &>/dev/null; then
    echo -e "${RED}❌ Lỗi: Không tìm thấy python3 trên máy chủ. Vui lòng kiểm tra PATH.${NC}"
    exit 1
fi

PY_VERSION=$(python3 --version)
echo -e "${GREEN}✓ Đã phát hiện:${NC} ${PY_VERSION}"

# 2. Khởi tạo môi trường ảo độc lập (.venv)
if [ ! -d ".venv" ]; then
    echo -e "${YELLOW}⚙️ Đang tạo môi trường ảo Python (.venv)...${NC}"
    python3 -m venv .venv
    echo -e "${GREEN}✓ Đã tạo thành công thư mục .venv${NC}"
else
    echo -e "${BLUE}ℹ️ Môi trường ảo .venv đã tồn tại sẵn.${NC}"
fi

# Kích hoạt virtual environment
# shellcheck disable=SC1091
source .venv/bin/activate
echo -e "${GREEN}✓ Đã kích hoạt môi trường ảo (.venv)${NC}"

# 3. Nâng cấp pip nội bộ trong .venv
echo -e "\n${YELLOW}📦 Đang nâng cấp pip...${NC}"
pip install --upgrade pip

# 4. Thiết lập thư mục Hugging Face Cache
if [ -z "${HF_HOME:-}" ]; then
    DEFAULT_CACHE_DIR="$(pwd)/.cache/huggingface"
    mkdir -p "${DEFAULT_CACHE_DIR}"
    export HF_HOME="${DEFAULT_CACHE_DIR}"
    echo -e "${GREEN}✓ Đã thiết lập HF_HOME cục bộ:${NC} ${HF_HOME}"
else
    echo -e "${GREEN}✓ Sử dụng HF_HOME đã cấu hình trước:${NC} ${HF_HOME}"
fi

# Đảm bảo các thư mục kết quả đã tồn tại
mkdir -p results/figures data/audio data/ground_truth

# 5. Cài đặt các thư viện phụ thuộc từ requirements.txt
echo -e "\n${YELLOW}📦 Đang cài đặt thư viện từ requirements.txt...${NC}"
pip install -r requirements.txt
echo -e "${GREEN}✓ Cài đặt dependencies thành công!${NC}"

# 6. Kiểm tra tính tương thích CUDA của PyTorch trong .venv
echo -e "\n${YELLOW}🔍 Kiểm tra khả năng hỗ trợ CUDA của PyTorch trong .venv...${NC}"
CUDA_CHECK_OUTPUT=$(python -c "
import sys
try:
    import torch
    avail = torch.cuda.is_available()
    cnt = torch.cuda.device_count() if avail else 0
    print(f'CUDA_AVAIL={avail}')
    print(f'CUDA_COUNT={cnt}')
    print(f'TORCH_VER={torch.__version__}')
    if avail:
        print(f'GPU_NAME={torch.cuda.get_device_name(0)}')
except Exception as e:
    print(f'ERROR={e}')
")

echo "${CUDA_CHECK_OUTPUT}" | while IFS= read -r line; do
    echo "  • ${line}"
done

# Kiểm tra nếu GPU server có nvidia-smi nhưng PyTorch trong venv chưa kích hoạt được CUDA
if command -v nvidia-smi &>/dev/null; then
    IS_CUDA_AVAIL=$(echo "${CUDA_CHECK_OUTPUT}" | grep "CUDA_AVAIL=True" || true)
    if [ -z "${IS_CUDA_AVAIL}" ]; then
        echo -e "\n${YELLOW}${BOLD}⚠️ CẢNH BÁO TƯƠNG THÍCH PHẦN CỨNG:${NC}"
        echo -e "Hệ thống có GPU NVIDIA nhưng bản PyTorch vừa cài đặt là bản CPU-only hoặc chưa khớp CUDA driver."
        echo -e "Để kích hoạt gia tốc GPU tối đa cho benchmark, hãy chạy lệnh sau trong .venv:"
        echo -e "${CYAN}${BOLD}  pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121${NC}"
        echo -e "(Lưu ý: Không can thiệp driver hệ thống hay cài đặt global, chỉ cài wheel trong .venv)\n"
    else
        echo -e "\n${GREEN}${BOLD}✓ PyTorch đã nhận diện GPU thành công! Gia tốc CUDA sẵn sàng.${NC}"
    fi
fi

# 7. Khởi tạo dấu vân tay môi trường ban đầu
echo -e "\n${YELLOW}📋 Đang thu thập và lưu vết Environment Fingerprint...${NC}"
python -c "
from common.result_schema import EnvironmentFingerprint
env = EnvironmentFingerprint.capture(environment_id='server_init')
env.save_to_json('results/experiment_environment.json')
print('  • Đã lưu kết quả tại: results/experiment_environment.json')
"

echo -e "\n${CYAN}${BOLD}==============================================================================${NC}"
echo -e "${GREEN}${BOLD}🎉 THIẾT LẬP MÔI TRƯỜNG HOÀN TẤT!${NC}"
echo -e "Để kích hoạt môi trường làm việc thủ công bất kỳ lúc nào:"
echo -e "${CYAN}${BOLD}  source .venv/bin/activate${NC}"
echo -e "${CYAN}${BOLD}==============================================================================${NC}\n"
