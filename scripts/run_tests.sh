#!/usr/bin/env bash
# ==============================================================================
# Script: run_tests.sh
# Mục đích: Chạy toàn bộ bộ kiểm thử tự động (Unit Tests & Integration Tests)
# Ghi chú:
# - Kiểm tra toàn vẹn các module audio, metrics, resource monitor, schema, text norm
# - Đảm bảo môi trường sẵn sàng trước khi thực thi benchmark tải trọng nặng
# ==============================================================================

set -euo pipefail

# Màu hiển thị
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
BOLD='\033[1m'
NC='\033[0m'

# Kích hoạt .venv nếu chưa được kích hoạt
if [ -z "${VIRTUAL_ENV:-}" ] && [ -f ".venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

echo -e "\n${CYAN}${BOLD}==============================================================================${NC}"
echo -e "${CYAN}${BOLD}    DUT AI GPU SERVER - CHẠY BỘ KIỂM THỬ CHUẨN HÓA (MEETLY STT TESTS)         ${NC}"
echo -e "${CYAN}${BOLD}==============================================================================${NC}\n"

echo -e "${YELLOW}🧪 Đang thực thi pytest...${NC}"
python -m pytest tests -v "$@"

echo -e "\n${GREEN}${BOLD}✓ Toàn bộ kiểm thử đã vượt qua thành công!${NC}\n"
