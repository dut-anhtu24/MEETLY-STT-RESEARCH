# Meetly STT Research Benchmark Framework

Framework đo kiểm thực nghiệm và đánh giá định lượng cho **Subsystem Nhận dạng Giọng nói (STT - Speech-to-Text)** của dự án **Meetly**.

Nghiên cứu này phục vụ 2 chế độ độc lập:
1. **Offline Mode**: Chuyển đổi video/cuộc họp ghi hình thành transcript có cấu trúc (ưu tiên Accuracy, Throughput, Resource, Long-form stability).
2. **Streaming Mode**: Chuyển đổi luồng âm thanh trực tiếp (livestream/meeting) thành partial & stable transcript thời gian thực (ưu tiên Latency, RTF < 1, Transcript stability, Hardware footprint).

---

## Cấu Trúc Dự Án

```text
meetly-stt-research/
├── configs/                            # Cấu hình YAML (models, pipelines, benchmark)
│   ├── models.yaml                     # Danh mục model và metadata backends
│   ├── offline_pipelines.yaml          # Cấu hình pipeline offline (VAD, timestamps)
│   ├── streaming_pipelines.yaml        # Cấu hình streaming (frame, buffer, agreement)
│   └── benchmark.yaml                  # Cấu hình thực nghiệm và danh mục thuật ngữ IT
├── data/                               # Dữ liệu phục vụ đo kiểm
│   ├── metadata.csv                    # Danh mục Tier A (Clean, Noisy, Meeting, Code-Switch, Silence, Long-form)
│   ├── prepare_dataset.py              # Script chuẩn bị và xác thực dataset
│   ├── audio/                          # File WAV chuẩn 16kHz Mono
│   └── ground_truth/                   # File text đối chứng
├── common/                             # Các module mã nguồn dùng chung
│   ├── audio.py                        # Đọc audio 16kHz, sinh frame, trích xuất video
│   ├── text_normalization.py          # Chuẩn hóa tiếng Việt & phân tích thuật ngữ IT
│   ├── metrics.py                      # WER, CER, RTF, Throughput, Revision Distance
│   ├── resource_monitor.py             # Giám sát Baseline/Peak/Delta RAM & VRAM (NVML/PyTorch)
│   ├── model_registry.py               # Skeleton đăng ký model & cách ly smoke-test
│   ├── result_schema.py                # Pydantic schemas, OOM handling & EnvironmentFingerprint
│   └── vad.py                          # Silero VAD wrapper với fallback an toàn
├── notebooks/                          # Notebooks thực nghiệm & visualization (Tiếng Việt)
│   ├── 01_model_screening.ipynb        # Sàng lọc ứng viên (CP2)
│   ├── 02_offline_pipeline_benchmark.ipynb # Benchmark pipeline offline (CP3)
│   └── 03_streaming_pipeline_benchmark.ipynb # Benchmark pipeline streaming (CP4)
├── results/                            # Lưu trữ kết quả thực nghiệm
│   ├── experiment_environment.json     # Fingerprint môi trường phần cứng
│   ├── model_screening.jsonl           # Source of Truth dạng JSON Lines
│   └── figures/                        # Đồ thị Pareto frontier tự động
├── scripts/                            # Bộ script thực thi server GPU & headless CLI
│   ├── check_environment.sh            # Kiểm tra phần cứng GPU, VRAM, RAM, CUDA, disk
│   ├── setup_server_env.sh             # Khởi tạo .venv độc lập và cấu hình HF_HOME
│   ├── run_tests.sh                    # Chạy kiểm thử tự động
│   ├── run_pipeline.py                 # CLI điều phối thực nghiệm (screening, offline, streaming)
│   ├── run_model_screening.sh          # Wrapper chạy sàng lọc mô hình
│   ├── run_offline_benchmark.sh        # Wrapper chạy benchmark offline
│   ├── run_streaming_benchmark.sh      # Wrapper chạy benchmark streaming
│   └── run_all_benchmarks.sh           # Chạy toàn bộ chuỗi thực nghiệm tuần tự
├── tests/                              # Bộ unit test kiểm thử nền tảng
└── docs/                               # Tài liệu nghiên cứu kỹ thuật & hướng dẫn server
```

---

## Hướng Dẫn Cài Đặt & Chạy Kiểm Thử

### 1. Cài đặt thư viện phụ thuộc
```bash
pip install -r requirements.txt
```

### 2. Chạy bộ kiểm thử tự động (Unit Tests)
```bash
python -m pytest tests
```

Toàn bộ 18 test cases kiểm thử audio, text normalization, metrics (WER/CER/RTF/Term Recall), resource monitor và Pydantic schema đều phải vượt qua 100%.

### 3. Tạo dữ liệu mẫu (Dummy Fixtures)
```bash
python data/prepare_dataset.py
```

### 4. Thu thập vân tay môi trường (Environment Fingerprint)
```bash
python -c "from common.result_schema import EnvironmentFingerprint; env = EnvironmentFingerprint.capture(); env.save_to_json('results/experiment_environment.json')"
```

---

## Thực Thi Trên DUT AI GPU Server

Xem tài liệu chi tiết tại: [docs/GPU_SERVER_EXECUTION.md](file:///docs/GPU_SERVER_EXECUTION.md).

Quy trình tóm tắt sau khi kết nối SSH:
```bash
# 1. Kiểm tra môi trường server (read-only, không yêu cầu sudo)
bash scripts/check_environment.sh

# 2. Thiết lập môi trường ảo độc lập (.venv) & cấu hình HF_HOME
bash scripts/setup_server_env.sh

# 3. Kích hoạt môi trường
source .venv/bin/activate

# 4. Chạy kiểm thử tự động
bash scripts/run_tests.sh

# 5. Chạy toàn bộ chuỗi benchmark (hoặc từng script riêng lẻ)
bash scripts/run_all_benchmarks.sh
```

---

## Quy Ước Ngôn Ngữ & Triển Khai
- **Tài liệu, Markdown cells, Docstring, Code comments**: Tiếng Việt.
- **Tên file, class, function, schema & config keys**: English.
- **Model siêu nhẹ (`whisper-tiny`)**: Chỉ dùng riêng cho unit test và smoke test, tuyệt đối không đưa vào bảng kết quả nghiên cứu chính.
