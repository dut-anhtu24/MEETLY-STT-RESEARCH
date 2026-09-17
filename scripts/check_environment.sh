#!/usr/bin/env bash
# ==============================================================================
# Script: check_environment.sh
# Mục đích: Kiểm tra và thu thập thông số môi trường phần cứng và phần mềm trên GPU Server
# Ghi chú:
# - Script CHỈ ĐỌC (read-only), tuyệt đối KHÔNG can thiệp hay sửa đổi cấu hình hệ thống
# - KHÔNG yêu cầu quyền sudo/root
# - Thu thập: hostname, OS, CPU, RAM, GPU, VRAM, NVIDIA driver, CUDA, Python, FFmpeg, disk
# ==============================================================================

set -uo pipefail

# Định nghĩa màu hiển thị ANSI cho giao diện terminal
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

echo -e "\n${CYAN}${BOLD}==============================================================================${NC}"
echo -e "${CYAN}${BOLD}    DUT AI GPU SERVER - KIỂM TRA MÔI TRƯỜNG HỆ THỐNG (MEETLY STT BENCHMARK)   ${NC}"
echo -e "${CYAN}${BOLD}==============================================================================${NC}\n"

# 1. Thông tin Hostname & Hệ điều hành (OS)
echo -e "${YELLOW}${BOLD}[1/8] Thông tin Máy chủ & Hệ điều hành:${NC}"
echo -e "  • Hostname: $(hostname 2>/dev/null || uname -n)"
if [ -f /etc/os-release ]; then
    OS_NAME=$(grep -E '^PRETTY_NAME=' /etc/os-release | cut -d= -f2 | tr -d '"')
    echo -e "  • Hệ điều hành: ${OS_NAME}"
else
    echo -e "  • Hệ điều hành: $(uname -s) $(uname -r) ($(uname -m))"
fi
echo -e "  • Kernel: $(uname -r)"

# 2. Thông tin Vi xử lý (CPU)
echo -e "\n${YELLOW}${BOLD}[2/8] Thông tin CPU & Luồng xử lý:${NC}"
if command -v lscpu &>/dev/null; then
    CPU_MODEL=$(lscpu | grep -E "Model name:" | sed 's/Model name:[ \t]*//')
    CPU_CORES=$(lscpu | grep -E "^CPU\(s\):" | sed 's/CPU(s):[ \t]*//')
    echo -e "  • Model CPU: ${CPU_MODEL}"
    echo -e "  • Số luồng CPU (Logical Cores): ${CPU_CORES}"
elif [ -f /proc/cpuinfo ]; then
    CPU_MODEL=$(grep -m 1 "model name" /proc/cpuinfo | cut -d: -f2 | sed 's/^[ \t]*//')
    CPU_CORES=$(grep -c "^processor" /proc/cpuinfo)
    echo -e "  • Model CPU: ${CPU_MODEL}"
    echo -e "  • Số luồng CPU: ${CPU_CORES}"
else
    echo -e "  • CPU: Không thể trích xuất thông tin qua lscpu/cpuinfo"
fi

# 3. Thông tin Bộ nhớ RAM hệ thống
echo -e "\n${YELLOW}${BOLD}[3/8] Bộ nhớ RAM Hệ thống:${NC}"
if command -v free &>/dev/null; then
    free -h | awk 'NR==1{printf "  • %-10s %-10s %-10s %-10s\n", $1, $2, $3, $4} NR==2{printf "  • %-10s %-10s %-10s %-10s\n", $1, $2, $3, $4}'
else
    echo -e "  • Lệnh free không khả dụng."
fi

# 4. Thông tin Card đồ họa NVIDIA & VRAM
echo -e "\n${YELLOW}${BOLD}[4/8] Trạng thái GPU NVIDIA & VRAM:${NC}"
if command -v nvidia-smi &>/dev/null; then
    echo -e "  ${GREEN}✓ Đã tìm thấy nvidia-smi! Chi tiết card đồ họa:${NC}"
    # Trích xuất tên card, driver version, tổng dung lượng VRAM
    nvidia-smi --query-gpu=index,name,driver_version,memory.total,utilization.gpu --format=csv,noheader | while IFS=',' read -r idx name driver mem util; do
        echo -e "  • GPU [${idx}]:${BOLD}${name}${NC} | VRAM:${BOLD}${mem}${NC} | Driver:${driver} | Tải GPU:${util}"
    done
else
    echo -e "  ${RED}✗ Không tìm thấy nvidia-smi trong PATH. Hệ thống có thể không có GPU NVIDIA hoặc driver chưa được nạp.${NC}"
fi

# 5. Thông tin CUDA & Bộ biên dịch NVCC
echo -e "\n${YELLOW}${BOLD}[5/8] Kiểm tra Bộ công cụ CUDA:${NC}"
if command -v nvcc &>/dev/null; then
    NVCC_VER=$(nvcc --version | grep "release" | sed 's/.*release //; s/,.*//')
    echo -e "  • CUDA Compiler (nvcc): ${GREEN}${NVCC_VER}${NC}"
else
    echo -e "  • CUDA Compiler (nvcc): ${YELLOW}Không tìm thấy nvcc (Chỉ có Driver CUDA Runtime hoặc trong venv)${NC}"
fi

# 6. Kiểm tra Python & Thư viện PyTorch CUDA
echo -e "\n${YELLOW}${BOLD}[6/8] Kiểm tra Môi trường Python & PyTorch:${NC}"
PYTHON_CMD="python3"
if [ -f ".venv/bin/python" ]; then
    PYTHON_CMD=".venv/bin/python"
    echo -e "  • Môi trường ảo (.venv): ${GREEN}Đã phát hiện .venv/bin/python${NC}"
fi

if command -v "$PYTHON_CMD" &>/dev/null; then
    PY_VER=$($PYTHON_CMD --version 2>&1)
    echo -e "  • Phiên bản Python: ${PY_VER}"
    
    # Kiểm tra torch và cuda qua python một dòng
    $PYTHON_CMD -c "
import sys
try:
    import torch
    print(f'  • PyTorch: {torch.__version__}')
    print(f'  • PyTorch CUDA Available: {\"${GREEN}True${NC}\" if torch.cuda.is_available() else \"${RED}False${NC}\"}')
    if torch.cuda.is_available():
        print(f'  • Số lượng GPU PyTorch thấy: {torch.cuda.device_count()}')
        for i in range(torch.cuda.device_count()):
            print(f'    - GPU {i}: {torch.cuda.get_device_name(i)} (VRAM: {torch.cuda.get_device_properties(i).total_memory / (1024**2):.0f} MB)')
    else:
        print('    (Cảnh báo: PyTorch đang chạy ở chế độ CPU-only)')
except ImportError:
    print('  • PyTorch: ${YELLOW}Chưa được cài đặt trong môi trường Python này${NC}')
" 2>/dev/null || echo -e "  • Không thể chạy kiểm tra PyTorch qua Python."
else
    echo -e "  ${RED}✗ Không tìm thấy python3!${NC}"
fi

# 7. Kiểm tra Công cụ Đa phương tiện FFmpeg
echo -e "\n${YELLOW}${BOLD}[7/8] Kiểm tra Công cụ FFmpeg (Xử lý âm thanh/video):${NC}"
if command -v ffmpeg &>/dev/null; then
    FF_VER=$(ffmpeg -version 2>&1 | head -n 1)
    echo -e "  ${GREEN}✓ ${FF_VER}${NC}"
else
    echo -e "  ${RED}✗ Không tìm thấy ffmpeg trong PATH. Cần cài đặt ffmpeg để tách audio từ video.${NC}"
fi

# 8. Dung lượng ổ đĩa & Cấu hình Hugging Face Cache
echo -e "\n${YELLOW}${BOLD}[8/8] Dung lượng Ổ đĩa & Thư mục Cache:${NC}"
df -h . | awk 'NR==1{printf "  • %-15s %-10s %-10s %-10s %-10s\n", $1, $2, $3, $4, $5} NR==2{printf "  • %-15s %-10s %-10s %-10s %-10s\n", $1, $2, $3, $4, $5}'

if [ -n "${HF_HOME:-}" ]; then
    echo -e "  • Biến môi trường HF_HOME: ${GREEN}${HF_HOME}${NC}"
else
    echo -e "  • Biến môi trường HF_HOME: ${YELLOW}Chưa set (Sẽ sử dụng thư mục cache mặc định ~/.cache/huggingface hoặc project local)${NC}"
fi

echo -e "\n${CYAN}${BOLD}==============================================================================${NC}"
echo -e "${GREEN}${BOLD}✓ Hoàn tất kiểm tra môi trường. Hệ thống đã sẵn sàng cấu hình hoặc chạy đo kiểm.${NC}"
echo -e "${CYAN}${BOLD}==============================================================================${NC}\n"
