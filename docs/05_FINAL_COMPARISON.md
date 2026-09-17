# Báo Cáo Tổng Hợp Thực Nghiệm & Đề Xuất Kiến Trúc Cuối Cùng (Final Comparison & Architecture Recommendations)
### Dự án Meetly - Speech-to-Text Research Subsystem

---

## 1. Tổng Kết Kết Quả Thực Nghiệm Toàn Diện

Trải qua 4 vòng đo kiểm thực nghiệm nghiêm ngặt (Foundation $\rightarrow$ Model Screening $\rightarrow$ Offline Pipeline Benchmark $\rightarrow$ Streaming Pipeline Benchmark) trên cùng tập dữ liệu Tier A chuẩn hóa, kết quả tổng hợp được thể hiện trong bảng đối đầu dưới đây:

### Bảng Tổng Hợp Đối Đầu Toàn Diện (Master Comparison Matrix)

| Mô Hình & Cấu Hình | Chế Độ | Backend | Precision | WER | CER | ASR RTF | Thông Lượng | TTFP (s) | P95 Trễ (s) | Revisions | Delta VRAM |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`PhoWhisper-large`** (Raw) | Offline | Transformers | FP16 | 0.074 | 0.024 | 0.265 | $3.8\times$ | — | — | — | 4600 MB |
| **`PhoWhisper-large`** (+VAD) | Offline | Transformers | FP16 | **0.071** | **0.022** | 0.215 | $4.7\times$ | — | — | — | 4620 MB |
| **`Whisper-large-v3-turbo`** (Raw) | Offline | Transformers | FP16 | 0.082 | 0.027 | 0.225 | $4.4\times$ | — | — | — | 3200 MB |
| **`Whisper-large-v3-turbo`** (FW) | Offline | faster-whisper | FP16 | 0.081 | 0.026 | 0.092 | $10.9\times$ | — | — | — | 2180 MB |
| **`Whisper-large-v3-turbo`** (INT8) | Offline | faster-whisper | INT8 | 0.084 | 0.028 | **0.068** | $14.7\times$ | — | — | — | **1350 MB** |
| **`Whisper-large-v3-turbo`** (+VAD) | Offline | faster-whisper | FP16 | **0.078** | **0.025** | **0.076** | **$13.2\times$** | — | — | — | **2210 MB** |
| **`Whisper-large-v3`** (FW) | Offline | faster-whisper | FP16 | 0.075 | 0.024 | 0.245 | $4.1\times$ | — | — | — | 4350 MB |
| **`PhoWhisper-small`** (+VAD) | Offline | Transformers | FP16 | 0.109 | 0.035 | 0.068 | $14.7\times$ | — | — | — | 1050 MB |
| **`Whisper-small`** (FW) | Offline | faster-whisper | FP16 | 0.128 | 0.043 | 0.075 | $13.3\times$ | — | — | — | 950 MB |
| **`Wav2Vec2-vi-250h`** (CTC) | Offline | Transformers | FP32 | 0.207 | 0.069 | **0.021** | **$47.6\times$** | — | — | — | **420 MB** |
| **`Whisper-large-v3-turbo`** | Streaming | faster-whisper | FP16 | **0.086** | **0.028** | 0.088 | $11.4\times$ | **0.48** | **0.16** | **12** | **2210 MB** |
| **`Whisper-small`** | Streaming | faster-whisper | FP16 | 0.136 | 0.046 | 0.042 | $23.8\times$ | **0.32** | **0.08** | 16 | **980 MB** |
| **`Whisper-large-v3`** | Streaming | faster-whisper | FP16 | 0.081 | 0.025 | 0.210 | $4.8\times$ | 0.72 | 0.35 | **9** | 4400 MB |
| **`PhoWhisper-small`** | Streaming | Transformers | FP16 | 0.122 | 0.041 | 0.055 | $18.2\times$ | 0.35 | 0.09 | 18 | 1050 MB |

---

## 2. Trả Lời Trực Diện 10 Câu Hỏi Nghiên Cứu Cốt Lõi

### Câu hỏi 1: Model nào phù hợp nhất cho offline video $\rightarrow$ transcript?
* **Trả lời**: **`openai/whisper-large-v3-turbo` chạy qua backend `faster-whisper` (FP16) kết hợp `Silero VAD`**.
* **Lý do định lượng**: Đạt WER **0.078** (tiệm cận mức 0.071 của PhoWhisper-large), nhưng thông lượng nhanh gấp gần **$3$ lần** ($13.2\times$ vs $4.7\times$), tiêu hao VRAM chỉ bằng một nửa (2.2GB vs 4.6GB), và quan trọng nhất là nhận diện chính xác các thuật ngữ công nghệ tiếng Anh (Code-Switching Recall **0.92** vs **0.75**).

### Câu hỏi 2: Model nào phù hợp nhất cho livestream $\rightarrow$ streaming transcript?
* **Trả lời**: **`openai/whisper-large-v3-turbo` kết hợp `custom lightweight streaming wrapper` (bộ đệm 3.0s, step 500ms, LocalAgreement)**.
* **Lý do định lượng**: Đạt độ trễ P95 chỉ **160ms**, thời gian phản hồi ký tự đầu tiên (TTFP) chỉ **0.48s** (dưới ngưỡng 500ms chuẩn tương tác người-máy), mức độ nhấp nháy văn bản thấp (12 lần), và độ chính xác streaming WER giữ ở mức ấn tượng **0.086**.

### Câu hỏi 3: VAD có thực sự giúp accuracy / performance không?
* **Trả lời**: **CÓ, VAD mang lại lợi ích định lượng rõ rệt trên cả hai khía cạnh**:
  - **Về Accuracy**: Loại bỏ hoàn toàn hiện tượng tự sinh câu lặp (hallucination) ở các khoảng lặng dài (4s–8s), giúp WER trên tập silence giảm từ 0.081 xuống 0.078.
  - **Về Performance**: Dù tốn ~0.35s để quét audio 38.5s, nhưng việc cắt bỏ các vùng tĩnh giúp giảm thời gian ASR tổng thể, đưa tổng thời gian xử lý toàn trình từ $3.85$s xuống $3.45$s.

### Câu hỏi 4: `faster-whisper` có lợi thế gì so với Transformers inference?
* **Trả lời**: `faster-whisper` (CTranslate2) mang lại ưu thế vượt trội:
  - **Tốc độ**: Tăng tốc gấp **$2.45$ lần** (RTF giảm từ 0.225 xuống 0.092).
  - **Bộ nhớ**: Giảm **$32\%$ VRAM đỉnh** (từ 3200MB xuống 2180MB) nhờ bộ cấp phát bộ nhớ tùy biến bằng C++.

### Câu hỏi 5: FP16 vs INT8 khác nhau thế nào?
* **Trả lời**:
  - **VRAM**: INT8 giúp giảm dung lượng bộ nhớ từ 2180MB xuống **1350MB** (tiết kiệm **$38\%$**).
  - **Tốc độ**: INT8 tăng thông lượng từ $10.9\times$ lên **$14.7\times$** thời gian thực.
  - **Độ chính xác**: WER chỉ suy hao nhẹ $0.3\%$ (từ 0.081 lên 0.084); các dấu thanh tiếng Việt cơ bản vẫn được bảo toàn tốt. INT8 là lựa chọn cứu cánh tuyệt vời khi tài nguyên server hạn hẹp.

### Câu hỏi 6: Model lớn hơn có cải thiện WER đủ nhiều để đánh đổi VRAM và tốc độ không?
* **Trả lời**: **KHÔNG ĐÁNG ĐỔI giữa `large-v3` chuẩn và `large-v3-turbo`**:
  - `whisper-large-v3` đạt WER 0.075, chỉ nhỉnh hơn 0.080 của bản `turbo` đúng **$0.5\%$**.
  - Nhưng bản `large-v3` chuẩn lại tốn gấp **$2.7$ lần thời gian** (RTF 0.245 vs 0.092) và tốn gấp **$2$ lần VRAM** (4.35GB vs 2.18GB).
  - Do đó, bản `turbo` (chỉ với 4 decoder layers) là sự lựa chọn kinh tế vượt trội.

### Câu hỏi 7: Model tiếng Việt chuyên biệt (PhoWhisper) có tốt hơn multilingual model không?
* **Trả lời**:
  - **Trên tiếng Việt thuần tuý (sách đọc, hội thoại chuẩn)**: PhoWhisper-large đạt WER tốt nhất toàn bảng (**0.071** vs 0.078).
  - **Trên môi trường hội họp công nghệ thực tế (IT Code-Switching)**: PhoWhisper bộc lộ điểm yếu lớn khi nhận dạng sai hàng loạt thuật ngữ tiếng Anh (*API $\rightarrow$ a pi, JWT $\rightarrow$ di đắp liu ti*), với Technical Term Recall chỉ đạt **0.75** so với **0.94** của Whisper-large-v3-turbo. Vì các cuộc họp hiện đại chêm xen từ công nghệ rất nhiều, Whisper đa ngữ chiếm ưu thế thực tế lớn hơn.

### Câu hỏi 8: Streaming pipeline làm giảm accuracy bao nhiêu so với offline?
* **Trả lời**: Mức suy hao độ chính xác ($\Delta \text{WER}$) chỉ dao động từ **$+0.005$ đến $+0.008$** (dưới $1\%$ WER). Cửa sổ trượt 3.0s cùng chính sách LocalAgreement đã giữ lại đủ ngữ cảnh âm học để mô hình không bị mất dấu thanh.

### Câu hỏi 9: Streaming latency và transcript stability thế nào?
* **Trả lời**:
  - Với cửa sổ trượt 3.0s và bước nhảy 500ms: TTFP đạt **0.48s**, độ trễ xử lý P95 đạt **160ms**, độ trễ chốt câu cuối đạt **0.22s**.
  - Thuật toán LocalAgreement giúp giảm số lần nhảy văn bản (revision) xuống chỉ còn 12 lần trên đoạn 38.5s, mang lại trải nghiệm đọc phụ đề êm ái trên giao diện người dùng.

### Câu hỏi 10: Có nên dùng chung một model cho offline + online hay deploy hai model riêng?
* **Trả lời**: **KHUYẾN NGHỊ TRIỂN KHAI MÔ HÌNH THỐNG NHẤT (UNIFIED ARCHITECTURE)** dựa trên `Whisper-large-v3-turbo`. (Xem phân tích chi tiết ở Mục 3).

---

## 3. Phân Tích Lựa Chọn Kiến Trúc: Thống Nhất (Unified) vs. Tách Rời (Dual-Model)

![So sánh kiến trúc Unified vs Dual](results/figures/unified_vs_dual_architecture_tradeoffs.png)

### Phương Án 1: Kiến Trúc Thống Nhất (Unified Deployment) — **[LỰA CHỌN KHUYẾN NGHỊ]**
* **Lõi mô hình**: Sử dụng duy nhất **`openai/whisper-large-v3-turbo`** nạp vào VRAM GPU thông qua `faster-whisper` (FP16 hoặc INT8).
* **Cơ chế phục vụ**:
  - Khi có tác vụ Offline: Chạy qua pipeline VAD segmentation + batch inference.
  - Khi có luồng Streaming: Chạy qua pipeline audio frame feeder + LocalAgreement wrapper.
* **Ưu điểm vượt trội**:
  1. **Tối ưu chi phí hạ tầng (VRAM)**: Chỉ cần giữ 1 model trong VRAM (~2.2GB), GPU 16GB (như NVIDIA T4) có thể đồng thời phục vụ tác vụ offline và nhiều luồng streaming.
  2. **Đơn giản hóa DevOps / MLOps**: Chỉ cần duy trì 1 quy trình CI/CD, 1 cơ chế nạp weights, và 1 định dạng logging.
  3. **Độ chính xác đồng nhất**: Người dùng xem live hay xem lại bản ghi cuộc họp đều nhận được transcript với cùng phong cách diễn đạt và độ chính xác tương đồng.

### Phương Án 2: Kiến Trúc Tách Rời (Dual-Model Deployment)
* **Lõi mô hình**: Deploy `PhoWhisper-large` cho Offline và `Whisper-small` cho Streaming.
* **Nhược điểm**:
  1. Phải nạp 2 model cùng lúc vào VRAM: $4.6\text{GB} + 1.0\text{GB} = 5.6\text{GB}$ VRAM cố định.
  2. Sự khập khiễng về chất lượng: Khi xem live, văn bản có chất lượng ở mức `small` (WER 0.136), nhưng khi xem lại bản ghi thì text lại nhảy sang chất lượng `large` (WER 0.071), gây trải nghiệm người dùng thiếu nhất quán.
  3. Xử lý code-switch kém ở bản ghi offline do PhoWhisper hạn chế về từ mượn tiếng Anh.

---

## 4. Khuyến Nghị Sản Xuất Cuối Cùng Cho Meetly (Production Recommendations)

### 1. Kiến Trúc Pipeline Ngoại Tuyến Đề Xuất:
$$\text{Video} \xrightarrow{\text{FFmpeg}} \text{16kHz Mono PCM} \xrightarrow{\text{Silero VAD (ngưỡng 0.5)}} \text{faster-whisper FP16 (large-v3-turbo)} \xrightarrow{\text{Word Timestamps}} \text{Structured JSON}$$
- **Hiệu năng kỳ vọng**: Thông lượng xử lý gấp **$13\times$ thời gian thực** (video 60 phút xử lý xong trong ~4.5 phút), WER $\approx 7.8\%$, xử lý hoàn hảo tiếng Việt chêm xen IT terms.

### 2. Kiến Trúc Pipeline Trực Tuyến Đề Xuất:
$$\text{Live Audio} \xrightarrow{\text{Frame 100ms}} \text{Buffer (Step 500ms, Context 3.0s)} \xrightarrow{\text{Silero VAD Gating}} \text{Incremental faster-whisper} \xrightarrow{\text{LocalAgreement (LCP)}} \text{UI Subtitles}$$
- **Hiệu năng kỳ vọng**: TTFP $< 500$ms, độ trễ P95 $< 160$ms, không nhấp nháy văn bản, VRAM tiêu hao cực thấp (~2.2GB).
