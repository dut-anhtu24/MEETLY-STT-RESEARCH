# So Sánh Kiến Trúc Toàn Diện Các Họ Mô Hình STT (Model Architecture Comparison)

Tài liệu này cung cấp bảng đối đầu ma trận tổng quan và phân tích kỹ thuật chuyên sâu giữa 4 họ mô hình nhận dạng giọng nói (STT) được đưa vào nghiên cứu thực nghiệm trong dự án **Meetly**:
1. **OpenAI Whisper Large v3 Turbo** (`openai/whisper-large-v3-turbo`)
2. **VinAI PhoWhisper** (`vinai/PhoWhisper-large`, `vinai/PhoWhisper-small`)
3. **Wav2Vec2 Vietnamese 250h** (`nguyenvulebinh/wav2vec2-base-vietnamese-250h`)
4. **OpenAI Whisper Small** (`openai/whisper-small`)

---

## 1. Bảng Ma Trận So Sánh Kiến Trúc Kỹ Thuật

| Tiêu Chí Kỹ Thuật | Whisper Large v3 Turbo | PhoWhisper (Large / Small) | Wav2Vec2 Vietnamese 250h | Whisper Small |
| :--- | :--- | :--- | :--- | :--- |
| **Kiến Trúc Cốt Lõi** | Transformer Encoder-Decoder (Seq2Seq) | Transformer Encoder-Decoder (Seq2Seq) | Wav2Vec 2.0 Encoder + CTC Linear Head | Transformer Encoder-Decoder (Seq2Seq) |
| **Tổng Số Tham Số** | ~809M | ~1550M (Large) / ~244M (Small) | ~95M | ~244M |
| **Cấu Trúc Encoder** | 32 layers, 20 heads, $d_{\text{model}} = 1280$ | 32 layers (Large) / 12 layers (Small) | 12 transformer layers, 8 heads, $d_{\text{model}} = 768$ | 12 layers, 12 heads, $d_{\text{model}} = 768$ |
| **Cấu Trúc Decoder** | **4 layers**, 20 heads, $d_{\text{model}} = 1280$ | 32 layers (Large) / 12 layers (Small) | **Không có decoder** (Non-autoregressive) | 12 layers, 12 heads, $d_{\text{model}} = 768$ |
| **Không Gian Từ Vựng (`vocab_size`)** | **51,866** | 51,865 (PhoBERT / BPE mapping) | 110 (Ký tự tiếng Việt + CTC tokens) | 51,865 |
| **Đặc Trưng Đầu Vào** | 128 Mel-frequency bins (16 kHz audio) | 128 Mel bins (Large) / 80 Mel bins (Small) | **16 kHz Raw Waveform Audio** (không dùng Mel) | 80 Mel-frequency bins (16 kHz audio) |
| **Dữ Liệu Huấn Luyện / Tinh Chỉnh** | ~5M giờ đa ngữ & pseudo-labeled | Fine-tuned trên **844 giờ** tiếng Việt đa accent | Pretrain 13k giờ YouTube, fine-tune 250h VLSP | 680k giờ đa ngữ & giám sát |
| **Cơ Chế Giải Mã (Decoding)** | Tự hồi quy (Autoregressive Greedy / Beam) | Tự hồi quy (Autoregressive Greedy / Beam) | **CTC Non-autoregressive Greedy / Viterbi** | Tự hồi quy (Autoregressive Greedy / Beam) |
| **Bản Chất Streaming (`Streaming Nature`)** | **Offline Seq2Seq; có thể bọc bằng streaming wrapper** | **Offline Seq2Seq; có thể bọc bằng streaming wrapper** | **CTC non-autoregressive; thuận lợi cho chunked/incremental inference** | **Offline Seq2Seq; có thể bọc bằng streaming wrapper** |
| **Độ Dài Context Tối Đa** | 30 giây (Chunk cửa sổ trượt) | 30 giây (Chunk cửa sổ trượt) | Không bị giới hạn cứng 30s | 30 giây (Chunk cửa sổ trượt) |
| **Định Dạng Đầu Ra (Formatting)** | Văn bản đầy đủ dấu câu và viết hoa | Văn bản đầy đủ dấu câu và viết hoa | **Văn bản thô, không viết hoa, không dấu câu** | Văn bản đầy đủ dấu câu và viết hoa |
| **Khả Năng Chêm Xen (Code-Switching)** | Rất mạnh (Đa ngữ gốc, bảo toàn thuật ngữ IT) | Kém hơn trên thuật ngữ ngoại lai mới | Kém (Chỉ nhận dạng bảng chữ cái tiếng Việt) | Khá (Dễ nhầm lẫn từ mượn ngắn) |
| **Bản Quyền & Triển Khai (License)** | MIT (Thương mại tự do) | MIT (Thương mại tự do) | **CC BY-NC 4.0 (Phi thương mại)** | MIT (Thương mại tự do) |

---

## 2. Phân Tích Bản Chất Streaming Kỹ Thuật

Một trong những sai lầm phổ biến khi đánh giá các hệ thống ASR là gán nhãn *"Native Streaming"* cho các mô hình có tốc độ suy luận nhanh ($RTF < 1$). Trong kiến trúc hệ thống của Meetly, bản chất kỹ thuật được xác định rõ:

### 2.1. Nhóm Encoder-Decoder Tự Hồi Quy (Whisper Turbo, PhoWhisper, Whisper Small)
- **Bản chất**: `Offline Seq2Seq; có thể bọc bằng streaming wrapper`.
- **Cơ chế hoạt động**: Mô hình yêu cầu một đoạn âm thanh đầu vào có độ dài cố định hoặc động (thường là mel-spectrogram 80/128 bins với độ dài từ 1s đến 30s), sau đó Decoder sinh từng token theo thứ tự tự hồi quy (autoregressive).
- **Phương thức triển khai streaming**: Để phục vụ chế độ trực tuyến, hệ thống bắt buộc phải bọc mô hình bằng một **Streaming Pipeline Wrapper** bao gồm:
  1. *Audio Ring Buffer* (Bộ đệm vòng nhận frame 20ms–100ms).
  2. *VAD/VAC Gating* (Chỉ kích hoạt suy luận khi có tiếng nói).
  3. *Rolling Context Window* (Cửa sổ ngữ cảnh trượt, ví dụ: 3.0s context với 500ms step).
  4. *Chính sách ổn định tiền tố (Stable-Prefix Agreement Policy)* để phát hiện và cố định các từ đã thống nhất giữa các lần trượt liên tiếp, tránh hiện tượng nhảy chữ trên giao diện.

### 2.2. Nhóm CTC Không Tự Hồi Quy (Wav2Vec2 Vietnamese 250h)
- **Bản chất**: `CTC non-autoregressive; thuận lợi cho chunked/incremental inference`.
- **Cơ chế hoạt động**: Khối mã hóa Wav2Vec2 trích xuất đặc trưng âm học trực tiếp từ sóng âm (raw waveform) qua mạng tích chập (Convolutional Feature Encoder) và các khối Transformer, sau đó một Linear Head ánh xạ trực tiếp sang phân phối xác suất của từng ký tự trên từng khung thời gian (Connectionist Temporal Classification).
- **Lưu ý triển khai**: Mặc dù giải mã CTC diễn ra song song trên toàn bộ chuỗi khung âm thanh và không bị nghẽn bởi vòng lặp decoder, checkpoint `nguyenvulebinh/wav2vec2-base-vietnamese-250h` được huấn luyện ở chế độ toàn câu (sentence-level) với self-attention hai chiều (bidirectional). Do đó, mô hình này **không mặc nhiên là một hệ thống native streaming ASR** (như Emformer hay Zipformer vốn có causal masking), nhưng lại đặc biệt thuận lợi cho việc chia nhỏ âm thanh thành các đoạn ngắn (chunked processing) với độ trễ tính toán rất nhỏ.

---

## 3. Tổng Hợp Kết Quả Thực Nghiệm Đối Chứng (Meetly Empirical Data)

Bảng số liệu dưới đây tổng hợp các kết quả thực nghiệm có độ tin cậy cao nhất, tuân thủ nghiêm ngặt quy tắc truy vết 4 thành phần metadata:

| Mô Hình / Checkpoint | Backend | Precision | Môi Trường Thực Thi | Dataset / Subset | WER (%) | CER (%) | Throughput / Latency | Peak VRAM (MB) | Nguồn Dữ Liệu Thực Nghiệm |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **PhoWhisper-Small** | transformers | float32 | DUT AI GPU Server (Tesla V100) | `vivos_test` (19 mẫu) | **9.22%** | 4.57% | 0.7x (RTF 1.362) | ~1100 MB | `results/vivos_test_summary.csv` |
| **Wav2Vec2-vi-250h** | transformers | float32 | DUT AI GPU Server (Tesla V100) | `vivos_test` (19 mẫu) | **14.18%** | 5.99% | **21.4x** (RTF 0.047) | **~420 MB** | `results/vivos_test_summary.csv` |
| **Whisper-Small** | faster-whisper | int8 | DUT AI GPU Server (Tesla V100) | `vivos_test` (19 mẫu) | 27.11% | 14.13% | 2.3x (RTF 0.438) | ~950 MB | `results/vivos_test_summary.csv` |
| **PhoWhisper-Large** | transformers | float16 | Colab GPU (Tesla T4) | `codeswitch_tech` | **7.10%** | 2.20% | 4.65x (RTF 0.215) | 4620 MB | `offline_benchmark.csv` (`off_abl_006`) |
| **Whisper-Large-v3-Turbo** | faster-whisper | float16 | Colab GPU (Tesla T4) | `codeswitch_tech` | 8.10% | 2.60% | **10.87x** (RTF 0.092) | 2180 MB | `offline_benchmark.csv` (`off_abl_002`) |
| **Whisper-Large-v3-Turbo** | faster-whisper | int8 | Colab GPU (Tesla T4) | `codeswitch_tech` | 8.40% | 2.80% | **14.71x** (RTF 0.068) | **1350 MB** | `offline_benchmark.csv` (`off_abl_003`) |
| **Turbo + Streaming Wrapper** | faster-whisper | float16 | Colab GPU (Tesla T4) | Streaming Simulation | 8.60% | 2.80% | P95: **0.16s** / TTFP: 0.48s | 2210 MB | `streaming_benchmark.csv` (`str_001_whisper_turbo`) |
| **Small + Streaming Wrapper** | faster-whisper | float16 | Colab GPU (Tesla T4) | Streaming Simulation | 13.60% | 4.60% | P95: **0.08s** / TTFP: 0.32s | **980 MB** | `streaming_benchmark.csv` (`str_002_whisper_small`) |

---

## 4. Chi Tiết Hồ Sơ Từng Mô Hình

Để xem chi tiết cấu trúc tầng, cơ chế giải mã, nhật ký thực nghiệm và hướng dẫn triển khai của từng mô hình, tham khảo các tài liệu chuyên sâu:

- [Kiến Trúc Chi Tiết OpenAI Whisper Large v3 Turbo](./01_WHISPER_LARGE_V3_TURBO.md)
- [Kiến Trúc Chi Tiết VinAI PhoWhisper](./02_PHOWHISPER.md)
- [Kiến Trúc Chi Tiết Wav2Vec2 Vietnamese 250h](./03_WAV2VEC2_VIETNAMESE.md)
- [Kiến Trúc Chi Tiết OpenAI Whisper Small](./04_WHISPER_SMALL.md)
- [Báo Cáo Đề Xuất Toàn Diện Cho Meetly (Master Recommendation)](../06_TOP_MODELS_AND_PIPELINE_RECOMMENDATION.md)
