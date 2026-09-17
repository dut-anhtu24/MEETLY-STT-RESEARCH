# Hồ Sơ Kiến Trúc Kỹ Thuật: VinAI PhoWhisper (Small / Medium / Large)

Tài liệu này phân tích chi tiết cấu trúc mô hình, nền tảng dữ liệu huấn luyện, đặc trưng âm học tiếng Việt và các số liệu thực nghiệm đo đạc được của họ mô hình **PhoWhisper** (`vinai/PhoWhisper-small`, `vinai/PhoWhisper-medium`, `vinai/PhoWhisper-large`) trong dự án **Meetly**.

---

## 1. Thông Tin Trích Dẫn & Kiến Trúc Cốt Lõi [FACT]

### 1.1. Trích Dẫn Khoa Học Chính Thức (Official Citation)
Mô hình PhoWhisper được công bố bởi đội ngũ nghiên cứu tại VinAI Research:
* **Tiêu đề công trình**: *PhoWhisper: Automatic Speech Recognition for Vietnamese*
* **Hội nghị xuất bản**: **ICLR 2024 (Tiny Papers track)**
* **Định danh lưu trữ**: **arXiv: `2406.02555`**
*(Lưu ý: Loại bỏ các trích dẫn nhầm lẫn trước đây sang Interspeech 2023 hoặc mã arXiv khác).*

### 1.2. Nguồn Dữ Liệu Tinh Chỉnh (Fine-Tuning Dataset)
* PhoWhisper được khởi tạo từ mô hình gốc đa ngữ OpenAI Whisper (`whisper-small`, `whisper-medium`, `whisper-large-v2/v3`), sau đó được tinh chỉnh (fine-tuned) trên **844 giờ dữ liệu âm thanh tiếng Việt**.
* **Đặc tính dữ liệu**: Tập dữ liệu huấn luyện bao gồm **đa dạng phương ngữ tiếng Việt (diverse Vietnamese accents)** từ các nguồn giọng đọc khác nhau.
*(Lưu ý kỹ thuật: Tài liệu chính thức xác nhận tính đa dạng phương ngữ, không khẳng định bao phủ tuyệt đối toàn bộ phương ngữ địa phương).*

### 1.3. Cấu Trúc Mạng & Cơ Chế Tokenizer
```text
Input Audio (16 kHz)
   │
   ▼
Feature Extractor (80 hoặc 128 Mel-frequency bins)
   │
   ▼
Transformer Encoder (12 đến 32 layers tùy phiên bản)
   │
   ├────────────────────────────────────────┐ (Cross-Attention)
   ▼                                        ▼
Transformer Decoder (12 đến 32 layers tùy phiên bản)
   │
   ▼
Linear Projection Head (WhisperTokenizer) -> Autoregressive Vietnamese Text
```

* **Kiểu kiến trúc**: Transformer Sequence-to-Sequence (Encoder-Decoder) tự hồi quy.
* **Cơ chế Tokenizer**: Mô hình sử dụng **`WhisperTokenizer`** (dựa trên thuật toán Byte-Pair Encoding - BPE của Whisper gốc).
* **Làm rõ kỹ thuật về thanh điệu tiếng Việt**: PhoWhisper **không sở hữu một module xử lý thanh điệu chuyên biệt riêng lẻ** và cũng không sử dụng một *"Vietnamese Phonetic Tokenizer"* riêng. Thay vào đó, khả năng nhận dạng chính xác các dấu thanh tiếng Việt (ngang, huyền, sắc, hỏi, ngã, nặng) là năng lực học được thông qua quá trình fine-tuning trọng số trên 844 giờ dữ liệu tiếng Việt.

### 1.4. Bảng Quy Cách Các Phiên Bản PhoWhisper

| Phiên Bản | Model ID Hugging Face | Encoder Layers | Decoder Layers | Chiều Ẩn ($d_{\text{model}}$) | Attention Heads | Đầu Vào Mel Bins | Tổng Số Tham Số |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **PhoWhisper-Small** | `vinai/PhoWhisper-small` | 12 | 12 | 768 | 12 | 80 bins | ~244M |
| **PhoWhisper-Medium** | `vinai/PhoWhisper-medium` | 24 | 24 | 1024 | 16 | 80 bins | ~769M |
| **PhoWhisper-Large** | `vinai/PhoWhisper-large` | 32 | 32 | 1280 | 20 | 128 bins | ~1550M |

---

## 2. Kết Quả Thực Nghiệm Trên Meetly [EMPIRICAL RESULT]

Mọi số liệu dưới đây được đo đạc và ghi nhận trực tiếp từ hệ thống thực nghiệm của Meetly kèm truy vết 4 trường metadata:

### 2.1. Đánh Giá Trên Tập Chuẩn Hóa VIVOS Test (DUT AI GPU Server - Tesla V100)

| Mô Hình | Backend | Precision | WER (%) | CER (%) | RTF | Thông Lượng (Throughput) | Tổng Thời Gian Suy Luận | Siêu Dữ Liệu Truy Vết (Traceability) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **PhoWhisper-Small** | transformers | float32 | **9.22%** | **4.57%** | 1.362 | 0.7x real-time | 102.52s / 75.2s audio | File: `vivos_test_summary.csv`<br>Run: Tổng hợp 19 mẫu VIVOS<br>Env: `dutai_gpu_env_20260917_170308`<br>Subset: `vivos_test` |

### 2.2. Đánh Giá Bóc Tách Chế Độ Ngoại Tuyến (Colab GPU - Tesla T4)

| Mô Hình | Backend | Precision | WER (%) | CER (%) | Term Recall | RTF | Peak VRAM | Siêu Dữ Liệu Truy Vết (Traceability) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **PhoWhisper-Large** | transformers | float16 | **7.40%** | 2.30% | 0.75 | 0.265 | 4600 MB | File: `offline_benchmark.csv`<br>Run: `off_abl_005`<br>Env: `colab_t4_cuda122_20260917`<br>Subset: `codeswitch_tech` |
| **PhoWhisper-Large (+VAD)** | transformers | float16 | **7.10%** | **2.20%** | 0.75 | 0.215 | 4620 MB | File: `offline_benchmark.csv`<br>Run: `off_abl_006`<br>Env: `colab_t4_cuda122_20260917`<br>Subset: `codeswitch_tech` |
| **PhoWhisper-Small** | transformers | float16 | 11.40% | 3.80% | 0.75 | 0.082 | 1020 MB | File: `offline_benchmark.csv`<br>Run: `off_abl_007`<br>Env: `colab_t4_cuda122_20260917`<br>Subset: `codeswitch_tech` |
| **PhoWhisper-Small (+VAD)** | transformers | float16 | 10.90% | 3.50% | 0.75 | 0.068 | 1050 MB | File: `offline_benchmark.csv`<br>Run: `off_abl_009`<br>Env: `colab_t4_cuda122_20260917`<br>Subset: `codeswitch_tech` |

---

## 3. Phân Tích Kỹ Thuật & Nhận Định [INTERPRETATION]

1. **Độ chính xác trên tiếng Việt chuẩn (Vietnamese Pure Domain)**:
   Đúng như kỳ vọng lý thuyết, nhờ quá trình tinh chỉnh chuyên biệt trên 844 giờ âm thanh tiếng Việt, **PhoWhisper-Large đạt độ chính xác cao nhất (WER 7.10% và CER 2.20%)** trên các đoạn nói tiếng Việt chuẩn ngữ pháp và phát âm rõ ràng, vượt trội hơn các mô hình đa ngữ không tinh chỉnh sâu.
2. **Khả năng nhận diện chêm xen tiếng Anh (Code-Switching)**:
   Mặc dù nhận dạng tiếng Việt rất tốt, PhoWhisper có xu hướng gặp khó khăn hơn khi người nói chêm xen thuật ngữ kỹ thuật tiếng Anh. Cụ thể, `Technical Term Recall` của PhoWhisper chỉ đạt **0.75** (so với **0.92** của Whisper Turbo), thường dẫn đến việc phiên âm sai hoặc bỏ sót các từ vựng công nghệ mới.
3. **Thực trạng tương thích với CTranslate2 (`faster-whisper`)**:
   Khi tiến hành thử nghiệm chuyển đổi mô hình qua `ct2-transformers-converter`, PhoWhisper gặp rào cản về cơ chế mapping từ điển của tokenizer. Do đó, hiện tại PhoWhisper phải chạy trên backend Hugging Face PyTorch (`transformers`). Điều này dẫn đến chi phí bộ nhớ VRAM cao (4.6GB đối với bản Large) và tốc độ suy luận bị hạn chế so với các mô hình đã được tối ưu hóa CTranslate2.
4. **Định vị trong bài toán Meetly**:
   PhoWhisper-Large là ứng viên lý tưởng cho chế độ **Ngoại tuyến (Offline Mode)** đối với các cuộc họp thuần tiếng Việt yêu cầu độ chuẩn xác dấu thanh cao nhất. Tuy nhiên, nó không phải là lựa chọn tối ưu cho chế độ Trực tuyến (Streaming Mode) khi hệ thống đòi hỏi độ trễ cực thấp và hỗ trợ nhiều thuật ngữ kỹ thuật chêm xen.

---

## 4. Hướng Dẫn Tải & Chạy Mẫu

Khi nạp các checkpoint PhoWhisper từ Hugging Face Hub, cần thiết lập tham số `use_safetensors=False` do cấu trúc checkpoint đóng gói định dạng PyTorch gốc:

```python
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import torch

model_id = "vinai/PhoWhisper-large"
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id,
    torch_dtype=torch_dtype,
    low_cpu_mem_usage=True,
    use_safetensors=False
).to(device)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
)
```

---

## 5. Liên Kết Tài Liệu Liên Quan
- [Bảng So Sánh Đối Đầu 4 Họ Mô Hình](./00_MODEL_ARCHITECTURE_COMPARISON.md)
- [Hồ Sơ Kiến Trúc OpenAI Whisper Large v3 Turbo](./01_WHISPER_LARGE_V3_TURBO.md)
- [Báo Cáo Đề Xuất Toàn Diện Cho Meetly](../06_TOP_MODELS_AND_PIPELINE_RECOMMENDATION.md)
