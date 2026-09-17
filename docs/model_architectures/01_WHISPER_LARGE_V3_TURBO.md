# Hồ Sơ Kiến Trúc Kỹ Thuật: OpenAI Whisper Large v3 Turbo

Tài liệu này phân tích chi tiết cấu trúc mạng, cơ chế tối ưu hóa giải mã, các chỉ số thực nghiệm đo đạc được trên hệ thống Meetly và định hướng triển khai cho mô hình **`openai/whisper-large-v3-turbo`**.

---

## 1. Thông Tin Kiến Trúc Cốt Lõi [FACT]

Dựa trên file cấu hình `config.json` và tài liệu chính thức từ OpenAI / Hugging Face:

```text
Input Audio (16 kHz)
   │
   ▼
Feature Extractor (128 Mel-frequency filterbanks, 25ms window, 10ms hop)
   │
   ▼
Convolutional Stem (2x 1D-Conv layers, GELU, stride=2 -> 4x subsampling)
   │
   ▼
Transformer Encoder (32 Layers, d_model=1280, 20 Attention Heads, MLP dim=5120)
   │
   ├────────────────────────────────────────┐ (Cross-Attention)
   ▼                                        ▼
Transformer Decoder (4 Layers, d_model=1280, 20 Attention Heads, MLP dim=5120)
   │
   ▼
Linear Projection Head (vocab_size = 51866) -> Autoregressive Token Generation
```

### 1.1. Bảng Thông Số Cấu Hình Chi Tiết
* **Họ mô hình**: OpenAI Whisper Turbo (Tối ưu hóa từ Whisper Large v3).
* **Kiểu kiến trúc**: Sequence-to-Sequence (Seq2Seq) Transformer Encoder-Decoder.
* **Số lượng tầng Encoder**: **32 layers**.
* **Số lượng tầng Decoder**: **4 layers** (giảm từ 32 layers của bản `whisper-large-v3` gốc).
* **Chiều ẩn mô hình ($d_{\text{model}}$)**: **1280**.
* **Số lượng đầu chú ý (Attention Heads)**: **20 heads** (cho cả Self-Attention và Cross-Attention).
* **Kích thước tầng trung gian (Feed-Forward / MLP dim)**: **5120**.
* **Không gian từ vựng (`vocab_size`)**: **51,866 tokens** (bao gồm đa ngữ, timestamps và special tokens).
* **Đặc trưng âm học đầu vào**: **128 Mel bins** (tính toán từ âm thanh chuẩn 16.000 Hz).
* **Tổng số tham số**: **~809 triệu tham số** (~809M parameters).
* **Bản quyền**: MIT License (Cho phép tích hợp thương mại tự do).

### 1.2. Bản Chất Thiết Kế & Sự Đổi Mới Của Bản Turbo
* Trong kiến trúc Transformer Seq2Seq truyền thống của Whisper Large v3 (32 encoder layers + 32 decoder layers), khâu Encoder chỉ chạy 1 lần cho toàn bộ chuỗi âm thanh, trong khi khâu Decoder phải chạy lặp đi lặp lại từng bước để sinh từng token (Autoregressive Generation Loop). Do đó, Decoder chiếm phần lớn thời gian trễ của quá trình suy luận.
* Whisper Large v3 Turbo giữ nguyên toàn bộ 32 tầng Encoder (bảo toàn năng lực biểu diễn âm học đa ngữ sâu rộng) nhưng thu gọn Decoder xuống còn 4 tầng. Điều này giúp giảm thiểu 87.5% khối lượng tính toán trong vòng lặp tự hồi quy, tạo bước nhảy vọt về thông lượng xử lý mà không làm suy giảm nghiêm trọng chất lượng nhận dạng.

---

## 2. Kết Quả Thực Nghiệm Trên Meetly [EMPIRICAL RESULT]

Mọi kết quả dưới đây được đo đạc trực tiếp trên framework benchmark của Meetly, có đầy đủ siêu dữ liệu truy vết:

### 2.1. Đánh Giá Chế Độ Ngoại Tuyến (Offline Benchmark)

| Cấu Hình Pipeline | Precision | RTF | Thông Lượng (Throughput) | WER (%) | CER (%) | Technical Term Recall | Peak VRAM | Siêu Dữ Liệu Truy Vết (Traceability) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Transformers Pipeline | float16 | 0.225 | 4.44x | 8.20% | 2.70% | 0.92 | 3200 MB | File: `offline_benchmark.csv`<br>Run: `off_abl_001`<br>Env: `colab_t4_cuda122_20260917`<br>Subset: `codeswitch_tech` |
| **faster-whisper (CTranslate2)** | **float16** | **0.092** | **10.87x** | **8.10%** | **2.60%** | **0.92** | **2180 MB** | File: `offline_benchmark.csv`<br>Run: `off_abl_002`<br>Env: `colab_t4_cuda122_20260917`<br>Subset: `codeswitch_tech` |
| **faster-whisper (CTranslate2)** | **int8** | **0.068** | **14.71x** | **8.40%** | **2.80%** | **0.92** | **1350 MB** | File: `offline_benchmark.csv`<br>Run: `off_abl_003`<br>Env: `colab_t4_cuda122_20260917`<br>Subset: `codeswitch_tech` |
| faster-whisper + Silero VAD | float16 | 0.076 | 13.16x | 7.80% | 2.50% | 0.92 | 2210 MB | File: `offline_benchmark.csv`<br>Run: `off_abl_004`<br>Env: `colab_t4_cuda122_20260917`<br>Subset: `codeswitch_tech` |

### 2.2. Đánh Giá Chế Độ Trực Tuyến (Streaming Simulation Benchmark)
Sử dụng bộ mô phỏng luồng âm thanh kết hợp rolling context 3.0s, step size 500ms và `LocalAgreement-inspired stability policy`:

* **Thời gian sinh từ đầu tiên (TTFP)**: **0.48 giây**.
* **Độ trễ xử lý trung vị (P50 Latency)**: **0.11 giây** (110 ms).
* **Độ trễ xử lý phân vị 95 (P95 Latency)**: **0.16 giây** (160 ms).
* **Độ trễ chốt câu hoàn thiện (Finalization Latency)**: **0.22 giây**.
* **Độ trễ xác nhận tiền tố ổn định (Stable Prefix Delay)**: **1.15 giây**.
* **Tần suất sửa đổi văn bản (Revision Count)**: **12 lần** trên đoạn kiểm thử.
* **Streaming WER**: **8.60%** (mức suy hao $\Delta \text{WER}$ chỉ $+0.5\%$ so với offline).
* **Mức chiếm dụng VRAM tối đa (Peak VRAM)**: **2210 MB**.
* **Traceability**: File `results/streaming_benchmark.csv`, Run ID `str_001_whisper_turbo`, Environment ID `colab_t4_cuda122_20260917`.

---

## 3. Phân Tích Kỹ Thuật & Nhận Định [INTERPRETATION]

1. **Khả năng nhận diện chêm xen ngôn ngữ (Code-Switching)**:
   Trong các cuộc họp công nghệ tại Meetly, người tham gia thường xuyên sử dụng các thuật ngữ IT tiếng Anh (*API, JWT, microservices, Docker, sprint, backend*). Whisper Turbo thể hiện ưu thế vượt trội với `Technical Term Recall` đạt **0.92**, không bị hiện tượng ép phiên âm tiếng Việt sai lệch.
2. **Hiệu quả của Backend CTranslate2 (`faster-whisper`)**:
   Khi chuyển từ Transformers pipeline sang `faster-whisper` FP16, tốc độ tăng gấp **2.45 lần** (từ 4.44x lên 10.87x real-time) và VRAM giảm từ 3.2GB xuống 2.18GB. Khi áp dụng lượng tử hóa INT8, thông lượng đạt **14.71x real-time** với bộ nhớ VRAM chỉ 1.35GB trong khi WER chỉ tăng nhẹ từ 8.1% lên 8.4%.
3. **Cơ sở cho Kiến Trúc Thống Nhất (Unified Architecture Candidate)**:
   Whisper Turbo là ứng viên duy nhất trong benchmark thể hiện sự cân bằng lý tưởng trên mọi khía cạnh: vừa đạt chất lượng Offline tiệm cận PhoWhisper-Large (WER 8.1% vs 7.1%), vừa đạt độ trễ P95 đủ nhỏ (< 200ms) để phục vụ Streaming mượt mà.

---

## 4. Tối Ưu Hóa Trong Tương Lai (Future Optimization)

Các kỹ thuật sau được ghi nhận có tiềm năng cải thiện hiệu năng nhưng **chưa được benchmark trực tiếp** trong phạm vi nghiên cứu này:
* **KV-Cache Quantization (INT8 / FP8)**: Nén bộ nhớ đệm Key-Value của 4 tầng Decoder để tiết kiệm thêm bộ nhớ khi xử lý phiên họp kéo dài.
* **Speculative Decoding**: Sử dụng một mô hình nhỏ (như `whisper-tiny`) làm draft model để tăng tốc độ sinh token cho Whisper Turbo trên GPU cao cấp.
* **FlashAttention-2 / SDPA Kernel Tuning**: Tối ưu hóa ma trận tính toán Attention ở tầng Encoder 32 layers.

---

## 5. Liên Kết Tài Liệu Liên Quan
- [Bảng So Sánh Đối Đầu 4 Họ Mô Hình](./00_MODEL_ARCHITECTURE_COMPARISON.md)
- [Báo Cáo Đề Xuất Toàn Diện Cho Meetly](../06_TOP_MODELS_AND_PIPELINE_RECOMMENDATION.md)
