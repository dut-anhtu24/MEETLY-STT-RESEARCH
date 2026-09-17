#!/usr/bin/env bash
# ==============================================================================
# Script: run_all_benchmarks.sh
# Mục đích: Thực thi toàn bộ chuỗi thực nghiệm chuẩn hóa Meetly STT tuần tự
# Thứ tự thực thi:
# 1. Chạy bộ kiểm thử tự động (Unit & Integration Tests)
# 2. Sàng lọc mô hình (Model Screening)
# 3. Đo kiểm pipeline ngoại tuyến (Offline Benchmark)
# 4. Đo kiểm mô phỏng trực tuyến (Streaming Benchmark)
# 5. Tổng hợp báo cáo và đường dẫn kết quả
# ==============================================================================

set -euo pipefail

GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
BOLD='\033[1m'
NC='\033[0m'

echo -e "\n${CYAN}${BOLD}==============================================================================${NC}"
echo -e "${CYAN}${BOLD}    DUT AI GPU SERVER - CHẠY TOÀN BỘ CHUỖI BENCHMARK MEETLY STT              ${NC}"
echo -e "${CYAN}${BOLD}==============================================================================${NC}\n"

START_TIME=$(date +%s)

# 1. Chạy tests
echo -e "${YELLOW}${BOLD}[BƯỚC 1/4] Chạy kiểm thử tự động...${NC}"
bash scripts/run_tests.sh

# 2. Chạy model screening
echo -e "\n${YELLOW}${BOLD}[BƯỚC 2/4] Thực thi Sàng lọc mô hình (Model Screening)...${NC}"
bash scripts/run_model_screening.sh "$@"

# 3. Chạy offline benchmark
echo -e "\n${YELLOW}${BOLD}[BƯỚC 3/4] Thực thi Pipeline ngoại tuyến (Offline Benchmark)...${NC}"
bash scripts/run_offline_benchmark.sh "$@"

# 4. Chạy streaming benchmark
echo -e "\n${YELLOW}${BOLD}[BƯỚC 4/4] Thực thi Mô phỏng trực tuyến (Streaming Benchmark)...${NC}"
bash scripts/run_streaming_benchmark.sh "$@"

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo -e "\n${CYAN}${BOLD}==============================================================================${NC}"
echo -e "${GREEN}${BOLD}🎉 TOÀN BỘ CHIẾN DỊCH BENCHMARK ĐÃ HOÀN TẤT TRONG ${DURATION} GIÂY!${NC}"
echo -e "Các tệp kết quả chính đã được cập nhật:"
echo -e "  • results/experiment_environment.json   (Thông số phần cứng & thư viện)"
echo -e "  • results/model_screening.jsonl / .csv   (Dữ liệu sàng lọc mô hình)"
echo -e "  • results/offline_pipeline_benchmark.jsonl / .csv (Dữ liệu pipeline offline)"
echo -e "  • results/streaming_pipeline_benchmark.jsonl / .csv (Dữ liệu pipeline streaming)"
echo -e "  • results/stream_events.jsonl           (Chi tiết từng sự kiện streaming)"
echo -e "${CYAN}${BOLD}==============================================================================${NC}\n"
