# Hồ Sơ Kiến Trúc Kỹ Thuật: Wav2Vec2 Vietnamese 250h

Tài liệu này phân tích chi tiết kiến trúc không tự hồi quy (Non-Autoregressive CTC), cấu trúc mạng, cơ chế giải mã và các kết quả thực nghiệm đo đạc được của mô hình **`nguyenvulebinh/wav2vec2-base-vietnamese-250h`** trong dự án **Meetly**.

---

## 1. Thông Tin Kiến Trúc Cốt Lõi [FACT]

Dựa trên model card chính thức và cấu hình Hugging Face:

```text
Raw Audio Waveform (16 kHz PCM)
   │
   ▼
Convolutional Feature Encoder (7 blocks 1D-CNN, kernel 10..2, stride 5..2 -> 320x subsampling)
   │ (Khung đặc trưng âm học 25ms, bước nhảy 20ms -> 50 Hz frame rate)
   ▼
Feature Projection (Linear + LayerNorm -> d_model = 768)
   │
   ▼
Transformer Context Network (12 Layers, 8 Attention Heads, d_model = 768, MLP dim = 3072)
   │ (Self-Attention 2 chiều / Bidirectional Attention)
   ▼
CTC Linear Projection Head (110 Output Classes: Bảng ký tự tiếng Việt + CTC Blank + Special Tokens)
   │
   ▼
Non-Autoregressive CTC Decoding (Greedy / Viterbi / Beam Search)
```

### 1.1. Bảng Thông Số Cấu Hình Chi Tiết
* **Tác giả / Định danh nguồn**: `nguyenvulebinh` (Checkpoint: `nguyenvulebinh/wav2vec2-base-vietnamese-250h`).
* **Kiến trúc cốt lõi**: Wav2Vec 2.0 Base kết hợp CTC Linear Classification Head.
* **Tổng số tham số**: **~95 triệu tham số** (~95M parameters).
* **Đặc trưng đầu vào**: **Sóng âm thô (16 kHz Raw Waveform)**, không qua bước trích xuất Mel-spectrogram.
* **Tỷ lệ trích xuất khung (Subsampling)**: 7 tầng 1D-CNN giảm tần số mẫu xuống 50 khung/giây (mỗi khung biểu diễn 20ms âm thanh).
* **Cấu trúc Transformer**: 12 tầng mạng Transformer, 8 Attention Heads, $d_{\text{model}} = 768$.
* **Đầu ra CTC**: 110 lớp (bảng chữ cái tiếng Việt thường, dấu thanh và ký tự khoảng trắng).
* **Dữ liệu huấn luyện**:
  - *Tiền huấn luyện tự giám sát (Pretraining)*: ~13.000 giờ audio tiếng Việt từ YouTube.
  - *Tinh chỉnh có giám sát (Fine-tuning)*: 250 giờ dữ liệu tiếng Việt có gán nhãn từ cuộc thi VLSP (Vietnamese Language and Speech Processing).
* **Giấy phép sử dụng (License)**: **Creative Commons Attribution-NonCommercial 4.0 (CC BY-NC 4.0)** — Cấm sử dụng trực tiếp cho mục đích thương mại sinh lợi nếu chưa có thỏa thuận riêng với tác giả.

### 1.2. Bản Chất Giải Mã CTC & Vấn Đề Streaming Kỹ Thuật
* **Giải mã không tự hồi quy (Non-Autoregressive)**: Khác với kiến trúc Seq2Seq của Whisper (phải chạy decoder lặp đi lặp lại từng token), CTC chiếu trực tiếp chuỗi đặc trưng âm học qua một Linear layer để dự đoán phân phối xác suất ký tự đồng thời trên toàn bộ các khung thời gian. Nhờ loại bỏ hoàn toàn vòng lặp decoder, mô hình có tiềm năng đạt thông lượng xử lý cực cao.
* **Bản chất Streaming thực tế**:
  > *"CTC non-autoregressive decoding, phù hợp với chunked/incremental processing nhưng model checkpoint hiện tại không mặc nhiên là native streaming ASR system."*
* **Nguyên nhân kỹ thuật**: Mô hình sử dụng mạng Transformer Encoder với cơ chế Self-Attention 2 chiều (Bidirectional Attention) được huấn luyện trên toàn bộ câu nói. Checkpoint này không có mặt nạ nhân quả (causal masking) như các kiến trúc streaming thuần túy (ví dụ Emformer). Mặc dù vậy, do chi phí tính toán rất thấp (~95M params), mô hình rất thuận lợi để tích hợp vào các pipeline xử lý theo phân đoạn ngắn (chunked processing) hoặc gia tăng dần (incremental inference).

---

## 2. Kết Quả Thực Nghiệm Trên Meetly [EMPIRICAL RESULT]

Mọi kết quả được trích xuất trực tiếp từ các file dữ liệu trong thư mục `results/`:

### 2.1. Đánh Giá Trên Tập Chuẩn Hóa VIVOS Test (DUT AI GPU Server - Tesla V100)

| Mô Hình | Backend | Precision | Throughput | RTF | Tổng Thời Gian Suy Luận | WER (%) | CER (%) | Siêu Dữ Liệu Truy Vết (Traceability) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Wav2Vec2-vi-250h** | transformers | float32 | **21.4x** | **0.047** | **3.51s** / 75.2s audio | 14.18% | 5.99% | File: `results/vivos_test_summary.csv`<br>Run: Tổng hợp 19 mẫu VIVOS<br>Env: `dutai_gpu_env_20260917_170308`<br>Subset: `vivos_test` |

*Chi tiết các mẫu đo đạc tiêu biểu từ `results/vivos_test_benchmark_results.csv`*:
- Mẫu `VIVOSDEV01_R180` (3.94s audio): Thời gian suy luận **0.296s** (RTF 0.075), WER 11.11%, CER 3.23%.
- Mẫu `VIVOSDEV03_R082` (3.94s audio): Thời gian suy luận **0.162s** (RTF 0.041), WER 16.67%, CER 5.00%.
- Mẫu `VIVOSDEV08_129` (3.88s audio): Thời gian suy luận **0.183s** (RTF 0.047), WER 23.08%, CER 6.98%.

### 2.2. Đánh Giá Mức Tiêu Hao Tài Nguyên Phần Cứng
- **Chiếm dụng VRAM (Peak VRAM)**: **~420 MB** (đo trên môi trường GPU Tesla V100).
- **Chiếm dụng RAM tiến trình**: **~650 MB**.
- **Traceability**: File `results/vivos_test_summary.csv` và `results/model_screening.csv`.

---

## 3. Phân Tích Kỹ Thuật & Nhận Định [INTERPRETATION]

1. **Ưu thế tuyệt đối về tốc độ và tài nguyên**:
   Do kiến trúc CTC gọn nhẹ (~95M tham số) và giải mã non-autoregressive, mô hình xử lý 75.2 giây âm thanh VIVOS chỉ trong **3.51 giây** trên GPU Tesla V100, đạt thông lượng **21.4 lần thời gian thực**. Lượng VRAM chiếm dụng chỉ ~420MB, thấp nhất trong tất cả các mô hình được thử nghiệm.
2. **Hạn chế về định dạng văn bản đầu ra**:
   Đầu ra của giải mã CTC là chuỗi ký tự thô:
   - **Không có dấu câu** (`. , ? !`).
   - **Không viết hoa** (toàn bộ là chữ thường).
   - **Không phân tách câu theo ngữ nghĩa**.
   Nếu đưa vào hệ thống Meetly, hệ thống bắt buộc phải xây dựng thêm pipeline hậu xử lý (punctuation and capitalization restoration), điều này sẽ làm tăng thêm độ trễ và phức tạp hóa kiến trúc.
3. **Độ chính xác và độ nhạy trước thuật ngữ tiếng Anh (Code-Switching)**:
   Do bộ từ vựng CTC chỉ có 110 ký tự tiếng Việt chuẩn, mô hình hoàn toàn bất lực trước các từ mượn kỹ thuật tiếng Anh (*Docker, JWT, API*), dẫn đến lỗi chèn hoặc xóa ký tự nghiêm trọng khi người nói chêm xen tiếng Anh trong cuộc họp.
4. **Hạn chế về giấy phép thương mại (License Barrier)**:
   Giấy phép CC BY-NC 4.0 là một rào cản pháp lý lớn nếu Meetly được thương mại hóa dưới dạng sản phẩm SaaS. Do đó, mô hình này chỉ phù hợp làm baseline tham chiếu tốc độ hoặc trong các dịch vụ nội bộ phi thương mại.

---

## 4. Hướng Dẫn Nạp Mô Hình Mẫu

```python
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import torch

model_id = "nguyenvulebinh/wav2vec2-base-vietnamese-250h"
device = "cuda:0" if torch.cuda.is_available() else "cpu"

processor = Wav2Vec2Processor.from_pretrained(model_id)
model = Wav2Vec2ForCTC.from_pretrained(model_id).to(device)

# Suy luận với audio raw waveform 16kHz
def transcribe_wav2vec2(audio_waveform):
    inputs = processor(audio_waveform, sampling_rate=16000, return_tensors="pt").input_values.to(device)
    with torch.no_grad():
        logits = model(inputs).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]
    return transcription
```

---

## 5. Liên Kết Tài Liệu Liên Quan
- [Bảng So Sánh Đối Đầu 4 Họ Mô Hình](./00_MODEL_ARCHITECTURE_COMPARISON.md)
- [Hồ Sơ Kiến Trúc OpenAI Whisper Large v3 Turbo](./01_WHISPER_LARGE_V3_TURBO.md)
- [Báo Cáo Đề Xuất Toàn Diện Cho Meetly](../06_TOP_MODELS_AND_PIPELINE_RECOMMENDATION.md)
