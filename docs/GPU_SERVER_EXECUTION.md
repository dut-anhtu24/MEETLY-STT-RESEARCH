# Hướng Dẫn Thực Thi Meetly STT Benchmark Trên DUT AI GPU Server

Tài liệu này hướng dẫn chi tiết quy trình kết nối, thiết lập môi trường và thực thi toàn bộ chuỗi thực nghiệm đo kiểm Speech-to-Text (STT) của dự án **Meetly** trên GPU Server của DUT AI.

---

## 1. Cấu Hình SSH và Kết Nối Server

Server DUT AI sử dụng Cloudflare Zero Trust Access để bảo mật cổng SSH. Bạn cần cài đặt `cloudflared` trên máy trạm cá nhân trước khi kết nối.

### Cấu hình file `~/.ssh/config` trên máy cá nhân:

```sshconfig
Host dutai-gpu
    HostName sshbkmaker.dutai.io.vn
    User dutai
    ProxyCommand cloudflared access ssh --hostname %h
```

> **Lưu ý an toàn:** Tuyệt đối không lưu mật khẩu, API token hoặc SSH private key vào repository này.

### Lệnh kết nối vào server:

```bash
ssh dutai-gpu
```

---

## 2. Clone hoặc Cập Nhật Codebase

Sau khi đăng nhập vào server qua SSH:

```bash
# Nếu clone mới:
git clone <repository_url>
cd MEETLY-STT-RESEARCH

# Nếu đã có thư mục, cập nhật code mới nhất:
git pull origin main
```

---

## 3. Kiểm Tra Môi Trường Phần Cứng & Thư Viện

Trước khi thiết lập hoặc chạy tác vụ nặng, hãy chạy script kiểm tra hệ thống. Script này **chỉ đọc (read-only)** và không yêu cầu quyền `sudo`:

```bash
bash scripts/check_environment.sh
```

Script sẽ tự động thu thập và hiển thị:
- Hostname, Hệ điều hành Linux, Phiên bản Kernel.
- Thông tin CPU và số luồng phần cứng.
- Dung lượng RAM hệ thống (`free -h`).
- Card đồ họa NVIDIA, tổng VRAM, phiên bản Driver (`nvidia-smi`).
- Bộ biên dịch CUDA (`nvcc`).
- Phiên bản Python và hỗ trợ CUDA của PyTorch.
- Trạng thái công cụ xử lý đa phương tiện FFmpeg.
- Dung lượng ổ đĩa khả dụng (`df -h`).

---

## 4. Thiết Lập Môi Trường Ảo Python (.venv)

Chạy script tạo và cấu hình môi trường ảo riêng biệt:

```bash
bash scripts/setup_server_env.sh
```

Script thực hiện các bước sau:
1. Tạo môi trường ảo riêng biệt tại `.venv` (không cài đặt thư viện global).
2. Kích hoạt `.venv` và nâng cấp `pip`.
3. Thiết lập thư mục cache Hugging Face (`HF_HOME`) tại `.cache/huggingface` nội bộ project (hoặc nhận biến môi trường có sẵn).
4. Cài đặt các thư viện từ `requirements.txt`.
5. Kiểm tra khả năng nhận diện GPU của PyTorch:
   - Nếu phát hiện card NVIDIA nhưng PyTorch chưa kích hoạt CUDA (do bản wheel CPU mặc định), script sẽ đưa ra gợi ý lệnh cài wheel GPU chuẩn mà không can thiệp system driver:
     ```bash
     pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
     ```
6. Tự động ghi lại dấu vân tay môi trường ban đầu vào `results/experiment_environment.json`.

---

## 5. Kích Hoạt Môi Trường Thủ Công

Bất kỳ khi nào mở phiên làm việc mới trên server, hãy kích hoạt môi trường:

```bash
source .venv/bin/activate
```

*(Tùy chọn) Chỉ định GPU cụ thể nếu server có nhiều GPU:*

```bash
# Ví dụ: Chỉ định sử dụng GPU 0
export CUDA_VISIBLE_DEVICES=0
```

*(Tùy chọn) Tùy biến thư mục cache Hugging Face:*

```bash
# Mặc định project đã trỏ vào .cache/huggingface, bạn có thể chuyển sang ổ đĩa dữ liệu khác nếu cần
export HF_HOME="/data/cache/huggingface"
```

---

## 6. Chạy Bộ Kiểm Thử Tự Động (Tests)

Đảm bảo mọi module (Audio, Text Normalization, Metrics, Resource Monitor, Model Registry, Schema) hoạt động trơn tru trước khi chạy benchmark:

```bash
bash scripts/run_tests.sh
```

Kết quả mong đợi: `18 passed` mà không có lỗi.

---

## 7. Chạy Sàng Lọc Mô Hình (Model Screening - Checkpoint 2)

Thực hiện đo kiểm độ chính xác (WER/CER, Technical Term Recall/Accuracy) và hiệu năng (RTF, Throughput, VRAM Delta) trên 7 ứng viên STT:

```bash
# Chạy đo kiểm thực tế với GPU tự động nhận diện:
bash scripts/run_model_screening.sh

# (Tùy chọn) Chỉ định GPU ID cụ thể:
bash scripts/run_model_screening.sh --gpu-id 0

# (Tùy chọn) Chạy thử nghiệm nhanh (smoke/mock test) kiểm tra pipeline:
bash scripts/run_model_screening.sh --mock
```

Kết quả sẽ được xuất ra:
- `results/model_screening.jsonl` (Source of Truth dạng JSON Lines)
- `results/model_screening.csv` (Bảng tổng hợp để đọc nhanh)

---

## 8. Chạy Đo Kiểm Pipeline Ngoại Tuyến (Offline Benchmark - Checkpoint 3)

Thực hiện đo kiểm các biến thể Pipeline ngoại tuyến (Video/Audio $\rightarrow$ VAD $\rightarrow$ ASR $\rightarrow$ Timestamps):
- Baseline Transformers vs faster-whisper (FP16 / INT8).
- Đánh giá vai trò của Silero VAD trong việc chống ảo giác âm thanh và phân đoạn câu.

```bash
bash scripts/run_offline_benchmark.sh
```

Kết quả sẽ được xuất ra:
- `results/offline_pipeline_benchmark.jsonl`
- `results/offline_pipeline_benchmark.csv`

---

## 9. Chạy Mô Phỏng Trực Tuyến (Streaming Benchmark - Checkpoint 4)

Thực hiện mô phỏng luồng âm thanh thời gian thực theo từng frame 100ms/20ms:
- Đánh giá cơ chế ổn định LocalAgreement ($k=2$).
- Đo đạc các chỉ số trực tuyến cốt lõi: TTFP (Time to First Partial), P50/P95 processing latency, Finalization Latency, Stable Prefix Delay, Revision Count.

```bash
bash scripts/run_streaming_benchmark.sh
```

Kết quả sẽ được xuất ra:
- `results/streaming_pipeline_benchmark.jsonl`
- `results/streaming_pipeline_benchmark.csv`
- `results/stream_events.jsonl` (Lưu vết chi tiết từng sự kiện frame trực tuyến)

---

## 10. Chạy Toàn Bộ Chuỗi Thực Nghiệm Tuần Tự (All-in-One)

Để chạy toàn bộ từ bước kiểm thử đến cả 3 chiến dịch benchmark trong một lệnh duy nhất:

```bash
bash scripts/run_all_benchmarks.sh
```

*(Hoặc chạy nhanh ở chế độ mô phỏng kiểm tra):*
```bash
bash scripts/run_all_benchmarks.sh --mock
```

---

## 11. Xem và Phân Tích Kết Quả

Sau khi hoàn tất, toàn bộ kết quả nằm trong thư mục `results/`:

```text
results/
├── experiment_environment.json        # Dấu vân tay môi trường (GPU, CUDA, OS, RAM, versions)
├── model_screening.csv                # Bảng kết quả sàng lọc mô hình
├── model_screening.jsonl              # Chi tiết JSON Lines sàng lọc
├── offline_pipeline_benchmark.csv     # Bảng kết quả pipeline ngoại tuyến
├── offline_pipeline_benchmark.jsonl   # Chi tiết JSON Lines pipeline ngoại tuyến
├── streaming_pipeline_benchmark.csv   # Bảng kết quả mô phỏng streaming
├── streaming_pipeline_benchmark.jsonl # Chi tiết JSON Lines streaming
├── stream_events.jsonl                # Log sự kiện frame-by-frame của streaming
└── figures/                           # Thư mục biểu đồ trực quan hóa (PNG)
```

Bạn có thể xem trực tiếp các bảng CSV trên server bằng `column` hoặc `pandas`:

```bash
# Xem bảng sàng lọc mô hình định dạng cột đẹp:
column -s, -t < results/model_screening.csv | less -S

# Xem thông số môi trường đã ghi nhận:
cat results/experiment_environment.json
```

---

## 12. Cơ Chế Xử Lý Lỗi OOM & Mô Hình Không Tương Thích

Hệ thống được thiết kế với cơ chế tự bảo vệ an toàn:
1. **CUDA Out of Memory (OOM):** Nếu mô hình kích thước quá lớn vượt ngưỡng VRAM card màn hình, runner sẽ tự động bắt ngoại lệ `torch.cuda.OutOfMemoryError`, giải phóng bộ nhớ đệm (`torch.cuda.empty_cache()`), ghi nhận trạng thái `skipped_oom` vào bản ghi và **tiếp tục đo kiểm các mô hình khác mà không làm gián đoạn toàn bộ chiến dịch**.
2. **Unsupported Backend:** Với các mô hình chưa có checkpoint CTranslate2 (như PhoWhisper trên faster-whisper), hệ thống tự động ghi nhận `skipped_unsupported`.
3. **Download Trọng Số:** Thời gian tải mô hình từ Hugging Face Hub (`load_time_s`) được tách biệt hoàn toàn khỏi thời gian suy luận (`asr_inference_time_s`), đảm bảo tính công bằng tuyệt đối cho các phép đo RTF và Latency.
