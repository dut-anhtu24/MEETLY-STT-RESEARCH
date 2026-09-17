# Hồ Sơ Kiến Trúc Kỹ Thuật: OpenAI Whisper Small

Tài liệu này phân tích chi tiết thông số kiến trúc, hiệu năng thực nghiệm và vai trò của mô hình **`openai/whisper-small`** như một ứng viên kích thước nhỏ (compact candidate) phục vụ các kịch bản phần cứng hạn chế hoặc yêu cầu độ trễ tối thiểu trong dự án **Meetly**.

---

## 1. Thông Tin Kiến Trúc Cốt Lõi [FACT]

Dựa trên cấu hình kỹ thuật chuẩn từ OpenAI / Hugging Face:

```text
Input Audio (16 kHz)
   │
   ▼
Feature Extractor (80 Mel-frequency filterbanks, 25ms window, 10ms hop)
   │
   ▼
Convolutional Stem (2x 1D-Conv layers, GELU, stride=2 -> 4x subsampling)
   │
   ▼
Transformer Encoder (12 Layers, d_model=768, 12 Attention Heads, MLP dim=3072)
   │
   ├────────────────────────────────────────┐ (Cross-Attention)
   ▼                                        ▼
Transformer Decoder (12 Layers, d_model=768, 12 Attention Heads, MLP dim=3072)
   │
   ▼
Linear Projection Head (vocab_size = 51865) -> Autoregressive Token Generation
```

### 1.1. Bảng Thông Số Cấu Hình Chi Tiết
* **Họ mô hình**: OpenAI Whisper (Phiên bản tiêu chuẩn).
* **Kiểu kiến trúc**: Transformer Sequence-to-Sequence (Encoder-Decoder) tự hồi quy.
* **Số lượng tầng Encoder**: **12 layers**.
* **Số lượng tầng Decoder**: **12 layers**.
* **Chiều ẩn mô hình ($d_{\text{model}}$)**: **768**.
* **Số lượng đầu chú ý (Attention Heads)**: **12 heads** (ở cả Self-Attention và Cross-Attention).
* **Kích thước tầng trung gian (Feed-Forward / MLP dim)**: **3072**.
* **Không gian từ vựng (`vocab_size`)**: **51,865 tokens**.
* **Đặc trưng âm học đầu vào**: **80 Mel-frequency bins** (tính toán từ âm thanh 16.000 Hz).
* **Tổng số tham số**: **~244 triệu tham số** (~244M parameters).
* **Bản quyền**: MIT License (Tự do thương mại hóa).

### 1.2. Bản Chất Thiết Kế & Định Vị Kỹ Thuật
Khác với bản Whisper Large v3 hay Large Turbo (sử dụng 128 Mel bins và không gian ẩn lớn $d_{\text{model}} = 1280$), Whisper Small được thiết kế với kích thước gọn nhẹ (~244M tham số), sử dụng 80 Mel bins. Do số lượng phép tính cho mỗi token ít hơn nhiều so với các bản Large, Whisper Small là một ứng viên nhỏ hơn nhằm đánh đổi một phần độ chính xác nhận dạng để đổi lấy độ trễ xử lý thấp và mức tiêu hao tài nguyên phần cứng khiêm tốn.

---

## 2. Kết Quả Thực Nghiệm Trên Meetly [EMPIRICAL RESULT]

Mọi kết quả dưới đây được đo đạc trực tiếp trên hệ thống Meetly và có đầy đủ siêu dữ liệu truy vết:

### 2.1. Đánh Giá Trên Tập Chuẩn Hóa VIVOS Test (DUT AI GPU Server - Tesla V100)

| Mô Hình | Backend | Precision | WER (%) | CER (%) | RTF | Thông Lượng (Throughput) | Tổng Thời Gian Suy Luận | Siêu Dữ Liệu Truy Vết (Traceability) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Whisper-Small** | faster-whisper | int8 | 27.11% | 14.13% | 0.438 | 2.3x real-time | 32.89s / 75.2s audio | File: `vivos_test_summary.csv`<br>Run: Tổng hợp 19 mẫu VIVOS<br>Env: `dutai_gpu_env_20260917_170308`<br>Subset: `vivos_test` |

### 2.2. Đánh Giá Chế Độ Trực Tuyến (Streaming Simulation Benchmark - Colab GPU Tesla T4)
Đo đạc trên bộ mô phỏng trực tuyến với rolling context 3.0s, step size 500ms và `LocalAgreement-inspired stability policy`:

* **Thời gian sinh từ đầu tiên (TTFP)**: **0.32 giây** (320 ms).
* **Độ trễ xử lý trung vị (P50 Latency)**: **0.05 giây** (50 ms).
* **Độ trễ xử lý phân vị 95 (P95 Latency)**: **0.08 giây** (80 ms).
* **Độ trễ chốt câu hoàn thiện (Finalization Latency)**: **0.15 giây**.
* **Độ trễ xác nhận tiền tố ổn định (Stable Prefix Delay)**: **0.95 giây**.
* **Tần suất sửa đổi văn bản (Revision Count)**: **16 lần**.
* **Streaming WER**: **13.60%** (CER: 4.60%).
* **Mức chiếm dụng VRAM tối đa (Peak VRAM)**: **980 MB**.
* **Traceability**: File `results/streaming_benchmark.csv`, Run ID `str_002_whisper_small`, Environment ID `colab_t4_cuda122_20260917`.

---

## 3. Phân Tích Kỹ Thuật & Nhận Định [INTERPRETATION]

1. **Ưu thế về độ trễ tức thời trong kịch bản Streaming**:
   Với độ trễ P95 chỉ **80 ms** và TTFP chỉ **320 ms**, Whisper Small là mô hình Encoder-Decoder có tốc độ phản hồi nhanh nhất trong các bài đo streaming. Người dùng gần như thấy chữ xuất hiện tức thì theo nhịp nói mà không cảm nhận độ trễ giao diện.
2. **Tiết kiệm tài nguyên phần cứng (VRAM < 1GB)**:
   Mức chiếm dụng VRAM đỉnh chỉ **980 MB** (khi chạy FP16) và dưới 600 MB (khi chạy INT8) cho phép triển khai mô hình này trên các máy trạm cấu hình thấp, laptop văn phòng hoặc chia sẻ chung GPU với các tác vụ khác mà không lo nguy cơ tràn bộ nhớ (Out-Of-Memory).
3. **Sự đánh đổi về độ chính xác tiếng Việt**:
   Khi so sánh với các dòng mô hình lớn hoặc mô hình tinh chỉnh chuyên sâu:
   - Trên tập VIVOS test, Whisper Small (INT8) có WER lên tới 27.11% (so với 9.22% của PhoWhisper-Small).
   - Trên tập streaming mô phỏng, WER đạt 13.60% (kém hơn mức 8.60% của Whisper Turbo).
   Do đó, mô hình này không phù hợp cho các cuộc họp quan trọng đòi hỏi ghi chép biên bản chính xác tuyệt đối từng câu từ.
4. **Định vị trong hệ thống Meetly**:
   Whisper Small là **phương án dự phòng nhẹ (Lightweight Fallback Candidate)** lý tưởng trong trường hợp:
   - Hệ thống streaming gặp tải cao cần giảm áp lực GPU.
   - Triển khai cục bộ trên máy người dùng (Edge / Client-side STT) không có GPU rời.

---

## 4. Liên Kết Tài Liệu Liên Quan
- [Bảng So Sánh Đối Đầu 4 Họ Mô Hình](./00_MODEL_ARCHITECTURE_COMPARISON.md)
- [Hồ Sơ Kiến Trúc OpenAI Whisper Large v3 Turbo](./01_WHISPER_LARGE_V3_TURBO.md)
- [Báo Cáo Đề Xuất Toàn Diện Cho Meetly](../06_TOP_MODELS_AND_PIPELINE_RECOMMENDATION.md)
