# Tài Liệu Kỹ Thuật: Đánh Giá Các Ứng Viên STT Ngoại Tuyến (Offline STT Candidates)
### Dự án Meetly - Speech-to-Text Research Benchmark

---

## 1. Bối Cảnh & Bài Toán Xử Lý Ngoại Tuyến (Offline Mode)

Trong hệ thống Meetly, chế độ ngoại tuyến phục vụ bài toán:
$$\text{Video / Cuộc họp ghi hình} \longrightarrow \text{Văn bản cuộc họp có cấu trúc (Structured JSON Transcript)}$$

### Luồng xử lý toàn trình (End-to-End Pipeline):
```text
Video File (.mp4/.mkv)
  │
  ▼ (FFmpeg Audio Extraction)
16 kHz Mono PCM Audio
  │
  ▼ (Silero VAD Segmentation)
Danh sách Speech Segments (Loại bỏ khoảng lặng)
  │
  ▼ (ASR Engine: faster-whisper / Transformers)
Nhận dạng & Trích xuất Word / Segment Timestamps
  │
  ▼ (Transcript Assembly)
Structured JSON Transcript
```

### Các tiêu chí ưu tiên hàng đầu:
1. **Độ chính xác (Accuracy)**: WER và CER tiếng Việt phải đạt mức thấp nhất có thể.
2. **Thông lượng xử lý (Throughput)**: RTF $\ll 1.0$ (tương đương thông lượng $10\times - 15\times$ thời gian thực) để xử lý nhanh video dài 1-2 tiếng.
3. **Độ bền vững trước nhiễu & khoảng lặng**: Không bị ảo giác (hallucination), không lặp từ vô tận khi người tham gia im lặng.
4. **Hiệu quả tài nguyên (VRAM/RAM)**: Triển khai khả thi trên máy chủ GPU tiêu chuẩn (16GB - 24GB VRAM).

---

## 2. Hồ Sơ Chi Tiết Top 3 Ứng Viên Offline (Offline Finalists)

Dựa trên kết quả từ **Checkpoint 2 (Model Screening)**, 3 ứng viên xuất sắc nhất đại diện cho 3 trường phái kiến trúc:

| Thuộc Tính | OFF-1: `PhoWhisper-large` | OFF-2: `Whisper-large-v3-turbo` | OFF-3: `PhoWhisper-small` |
| :--- | :--- | :--- | :--- |
| **Định hướng** | Tối đa độ chính xác tiếng Việt | Cân bằng đa ngữ, tốc độ & code-switch | Tiết kiệm tài nguyên & suy luận nhẹ |
| **Nhà phát triển** | VinAI Research | OpenAI | VinAI Research |
| **Kiến trúc** | Seq2Seq (Encoder-Decoder) | Seq2Seq (Encoder 32 layers, Decoder 4 layers) | Seq2Seq (Encoder-Decoder) |
| **Số tham số** | ~1550 Triệu (1.55B) | ~809 Triệu (809M) | ~244 Triệu (244M) |
| **VRAM đỉnh (FP16)** | ~4600 MB | ~2180 MB (`faster-whisper`) | ~1020 MB |
| **Ngôn ngữ mục tiêu** | Tiếng Việt chuyên sâu | Đa ngữ (Tiếng Việt & Tiếng Anh) | Tiếng Việt |
| **Inductive Bias** | Tokenizer tinh chỉnh theo âm vị tiếng Việt | Dữ liệu huấn luyện 5 triệu giờ đa ngữ | Tokenizer tiếng Việt tối ưu kích thước |
| **Backend tối ưu** | HuggingFace PyTorch SDPA | faster-whisper (CTranslate2) | HuggingFace PyTorch SDPA |

---

## 3. Kết Quả Thử Nghiệm Bóc Tách Định Lượng (Ablation Study)

Các thí nghiệm bóc tách được thực hiện trên cùng tập dữ liệu chuẩn hóa Tier A với kết quả đo lường khách quan:

| Mã Thử Nghiệm | Mô Hình | Backend | Precision | VAD | WER | CER | ASR RTF | Thông Lượng | Delta VRAM |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `off_abl_001` | `whisper-large-v3-turbo` | Transformers | FP16 | ❌ | 0.082 | 0.027 | 0.225 | $4.4\times$ | 3200 MB |
| `off_abl_002` | `whisper-large-v3-turbo` | faster-whisper | FP16 | ❌ | 0.081 | 0.026 | 0.092 | $10.9\times$ | 2180 MB |
| `off_abl_003` | `whisper-large-v3-turbo` | faster-whisper | INT8 | ❌ | 0.084 | 0.028 | 0.068 | $14.7\times$ | 1350 MB |
| `off_abl_004` | `whisper-large-v3-turbo` | faster-whisper | FP16 | ✅ | 0.078 | 0.025 | 0.076 | $13.2\times$ | 2210 MB |
| `off_abl_005` | `PhoWhisper-large` | Transformers | FP16 | ❌ | 0.074 | 0.023 | 0.265 | $3.8\times$ | 4600 MB |
| `off_abl_006` | `PhoWhisper-large` | Transformers | FP16 | ✅ | 0.071 | 0.022 | 0.215 | $4.7\times$ | 4620 MB |
| `off_abl_007` | `PhoWhisper-small` | Transformers | FP16 | ❌ | 0.114 | 0.038 | 0.082 | $12.2\times$ | 1020 MB |
| `off_abl_008` | `PhoWhisper-small` | Transformers | FP16 | ✅ | 0.109 | 0.035 | 0.068 | $14.7\times$ | 1050 MB |

---

## 4. Phân Tích Các Phát Hiện Thực Nghiệm

### A. Tác Động Thực Tế Của Silero VAD
- **Triệt tiêu ảo giác (Hallucination Suppression)**: Trong tập `silence_pause` (chứa các đoạn ngắt quãng 4s-6s), mô hình Whisper chạy raw (không VAD) đôi khi tự sinh các chuỗi lặp vô nghĩa (ví dụ: *"cảm ơn các bạn đã lắng nghe cảm ơn..."*). Khi kích hoạt Silero VAD, các đoạn tĩnh bị loại bỏ trước khi đưa vào ASR, giúp **WER giảm từ 0.081 xuống 0.078**.
- **Tối ưu thời gian xử lý**: Mặc dù VAD tốn khoảng $0.35$s để quét audio 38.5s, nhưng do loại bỏ được ~20% thời lượng không có tiếng nói, thời gian ASR thuần túy giảm tương ứng, giúp tổng thời gian toàn trình giảm từ $3.85$s xuống $3.45$s.

### B. Lợi Thế Của `faster-whisper` (CTranslate2) So Với Transformers
- **Tốc độ**: Trên cùng mô hình `whisper-large-v3-turbo` (FP16), `faster-whisper` đạt RTF **0.092** ($10.9\times$ thời gian thực) so với Transformers RTF **0.225** ($4.4\times$ thời gian thực) $\rightarrow$ **Nhanh gấp ~2.45 lần**.
- **Bộ nhớ VRAM**: Giảm từ 3200 MB xuống 2180 MB ($\approx 32\%$ tiết kiệm) nhờ cơ chế quản lý bộ nhớ tùy biến của CTranslate2.

### C. Đánh Đổi Lượng Tử Hóa (Quantization Trade-off: FP16 vs INT8)
- `faster-whisper INT8` đạt RTF ấn tượng **0.068** ($14.7\times$ thời gian thực) và chỉ tiêu hao **1350 MB VRAM** (tiết kiệm ~38% so với FP16).
- Mức độ suy hao độ chính xác tiếng Việt là rất nhỏ: WER tăng từ 0.081 lên 0.084 ($\Delta \text{WER} = +0.003$), các dấu thanh chuẩn không bị méo mó nghiêm trọng.

### D. So Sánh Độc Lập Giữa Các Họ Mô Hình
- **Tiếng Việt hội thoại chuẩn**: `PhoWhisper-large` cho kết quả chính xác nhất (WER 0.071), vượt qua cả Whisper-large-v3-turbo.
- **Tiếng Việt chêm xen tiếng Anh (Code-Switching)**: `Whisper-large-v3-turbo` áp đảo hoàn toàn với Technical Term Recall đạt **0.92** và Accuracy đạt **0.90** (nhận diện chính xác *API, JWT, Spring Boot, backend*), trong khi PhoWhisper-large chỉ đạt Recall **0.75** (thường nhận dạng sai các từ tiếng Anh thành âm tiếng Việt tương tự).

---

## 5. Cấu Trúc Transcript Có Cấu Trúc (Structured JSON)

Kết quả cuối cùng của pipeline offline xuất ra định dạng JSON hoàn chỉnh lưu tại `results/sample_offline_transcript.json`:
```json
{
  "metadata": {
    "session_id": "meeting_20260917_tech_sync",
    "model_id": "openai/whisper-large-v3-turbo",
    "backend": "faster-whisper",
    "precision": "float16",
    "vad_enabled": true,
    "audio_duration_s": 38.5,
    "asr_inference_time_s": 2.93,
    "total_processing_time_s": 3.45,
    "asr_rtf": 0.076
  },
  "segments": [
    {
      "id": 0,
      "start": 0.52,
      "end": 4.18,
      "text": "chào các bạn hôm nay team mình sẽ review kiến trúc Meetly",
      "words": [
        {"word": "chào", "start": 0.52, "end": 0.85},
        {"word": "các", "start": 0.88, "end": 1.05},
        {"word": "bạn", "start": 1.08, "end": 1.32},
        {"word": "Meetly", "start": 4.05, "end": 4.18}
      ]
    }
  ],
  "full_transcript": "chào các bạn hôm nay team mình sẽ review kiến trúc Meetly..."
}
```

---

## 6. Khuyến Nghị Sơ Bộ Cho Chế Độ Ngoại Tuyến

1. **Ứng viên sản xuất đề xuất chính (Primary Production Candidate)**:
   - **Pipeline**: `Video` $\rightarrow$ FFmpeg 16kHz PCM $\rightarrow$ `Silero VAD` $\rightarrow$ `Whisper-large-v3-turbo (faster-whisper FP16)` $\rightarrow$ Word Timestamps $\rightarrow$ Structured JSON.
   - **Lý do**: Cân bằng hoàn hảo giữa tốc độ ($13\times$ thời gian thực), khả năng xử lý thuật ngữ công nghệ tiếng Anh xuất sắc, VRAM chỉ ~2.2GB và loại bỏ triệt để ảo giác nhờ VAD.
2. **Ứng viên độ chính xác cao khi có GPU lớn (High-Accuracy Alternative)**:
   - `PhoWhisper-large + Silero VAD` cho các cuộc họp tiếng Việt 100% không chêm xen từ ngoại lai.
