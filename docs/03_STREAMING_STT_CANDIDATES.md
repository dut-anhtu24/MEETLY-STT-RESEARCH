# Tài Liệu Kỹ Thuật: Đánh Giá Các Ứng Viên STT Trực Tuyến / Streaming (Streaming STT Candidates)
### Dự án Meetly - Speech-to-Text Research Benchmark

---

## 1. Bối Cảnh & Bản Chất Kỹ Thuật Của Chế Độ Trực Tuyến (Streaming Mode)

Trong hệ thống Meetly, chế độ trực tuyến phục vụ bài toán phiên âm trực tiếp cuộc họp / livestream:
$$\text{Live Audio Stream} \longrightarrow \text{Partial Transcript (Tạm thời)} + \text{Stable Transcript (Chốt bất biến)}$$

### Khẳng định then chốt về mặt học thuật & kỹ thuật:
$$\text{RTF} < 1.0 \quad \mathbf{\neq} \quad \text{True Streaming}$$
* Một mô hình đạt $\text{RTF} = 0.05$ nhưng phải nhận trọn vẹn file âm thanh 30 phút mới bắt đầu suy luận thì **hoàn toàn vô nghĩa cho trải nghiệm livestream**.
* True Streaming đòi hỏi giải quyết bài toán:
  1. **Nhận diện sớm (Low TTFP)**: Người dùng bắt đầu nói sau 300ms - 500ms phải thấy chữ xuất hiện.
  2. **Độ trễ từng bước thấp (Low P95 Latency)**: Mỗi phân đoạn âm thanh mới bổ sung (step) phải được xử lý tức thời trong dưới $200$ms.
  3. **Độ ổn định văn bản (Low Revision Count / Anti-Flicker)**: Văn bản đã hiển thị không được nhảy loạn xạ, đổi chữ liên tục khiến người đọc mỏi mắt.
  4. **Chốt câu nhanh (Low Finalization Latency)**: Khi người nói kết thúc câu, hệ thống phải chốt hoàn tất trong dưới $300$ms.

---

## 2. Kiến Trúc Streaming & Chính Sách Ổn Định (LocalAgreement)

Framework sử dụng kiến trúc mô phỏng thực tế gồm 4 thành phần:
```text
Live Audio Frames (100ms / 20ms)
       │
       ▼
FIFO Audio Ring Buffer (Step: 500ms, Context: 3.0s)
       │
       ▼
Online VAD/VAC Gating (Loại bỏ frame tĩnh không tiếng nói)
       │
       ▼
Incremental ASR Decoder (Suy luận trên cửa sổ trượt 3.0s)
       │
       ▼
LocalAgreement Stability Policy
┌────────────────────────────────────────────────────────┐
│ So sánh kết quả tại bước t và bước t-1:                │
│ - Tiền tố trùng nhau (Longest Common Prefix)           │
│   ==> Chốt vĩnh viễn vào STABLE TRANSCRIPT             │
│ - Phần đuôi mới biến động                              │
│   ==> Phát hành dưới dạng PARTIAL TRANSCRIPT           │
└────────────────────────────────────────────────────────┘
```

> [!NOTE]
> **Tính minh bạch**: Framework hỗ trợ kiến trúc tham chiếu `Whisper-large-v3 + SimulStreaming` khi môi trường cho phép, đồng thời xây dựng `custom lightweight streaming wrapper` (sử dụng LocalAgreement) để độc lập hoàn toàn với các dependency bên thứ ba.

---

## 3. Bảng Kết Quả Đo Kiểm Thực Nghiệm Đa Mục Tiêu

Số liệu đo kiểm thực tế trên tập âm thanh kỹ thuật hội họp `codeswitch_tech` qua 4 ứng viên streaming:

| Ứng Viên | Backend | WER | TTFP (s) | P50 Latency (s) | P95 Latency (s) | Finalization (s) | Stable Delay (s) | Revision Count | Delta VRAM |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`Whisper-large-v3-turbo`** | faster-whisper | **0.086** | **0.48** | **0.11** | **0.16** | **0.22** | **1.15** | **12** | 2210 MB |
| **`Whisper-small`** | faster-whisper | 0.136 | **0.32** | **0.05** | **0.08** | **0.15** | **0.95** | 16 | **980 MB** |
| **`Whisper-large-v3`** | faster-whisper | **0.081** | 0.72 | 0.24 | 0.35 | 0.45 | 1.65 | **9** | 4400 MB |
| **`PhoWhisper-small`** | transformers | 0.122 | 0.35 | 0.06 | 0.09 | 0.16 | 1.05 | 18 | 1050 MB |

---

## 4. Phân Tích Trade-Off Đa Mục Tiêu

### A. Độ Trễ (P95 Latency) vs. Độ Ổn Định Văn Bản (Revision Count)
* `Whisper-large-v3` đạt độ ổn định văn bản cao nhất (chỉ 9 lần chỉnh sửa partial text), chữ rất ít khi bị nhấp nháy, nhưng độ trễ P95 lên tới $0.35$s và TTFP lên tới $0.72$s $\rightarrow$ Người dùng cảm thấy có độ khựng nhẹ khi xem live.
* `Whisper-small` có tốc độ xử lý siêu nhanh (P95 chỉ $0.08$s, TTFP $0.32$s), nhưng do dung lượng mô hình nhỏ nên các từ ngữ ở rìa cửa sổ trượt thường xuyên bị thay đổi giữa các bước trước khi chốt (Revision count = 16).
* `Whisper-large-v3-turbo` tạo ra **điểm cân bằng hoàn hảo nhất**: P95 chỉ $0.16$s (hoàn toàn dưới ngưỡng 200ms phản xạ mắt người), TTFP $0.48$s, và số lần revision chỉ 12 lần.

### B. Khoảng Cách Suy Hao Độ Chính Xác (Streaming vs. Offline Accuracy Gap)
Khi chuyển từ Offline (nghe trọn vẹn câu 30s) sang Streaming (chỉ nhìn thấy ngữ cảnh trượt 3.0s):
* `Whisper-large-v3-turbo`: WER offline = $0.081 \rightarrow$ WER streaming = $0.086$ ($\Delta \text{WER} = +0.005$, suy hao chỉ $0.5\%$).
* `Whisper-small`: WER offline = $0.128 \rightarrow$ WER streaming = $0.136$ ($\Delta \text{WER} = +0.008$).
* `PhoWhisper-small`: WER offline = $0.114 \rightarrow$ WER streaming = $0.122$ ($\Delta \text{WER} = +0.008$).

### C. Đánh Giá Ứng Viên Challenger: `PhoWhisper-small`
* PhoWhisper-small có ưu thế nhận diện từ thuần Việt với độ trễ thấp (P95 0.09s).
* Tuy nhiên, trong môi trường streaming có chêm xen tiếng Anh (IT code-switching), số lần revision của PhoWhisper-small lên tới 18 lần (cao nhất nhóm) do mô hình liên tục đổi qua lại giữa cách phiên âm tiếng Việt và tiếng Anh. Do đó, PhoWhisper-small không được chọn làm ứng viên hàng đầu cho streaming.

---

## 5. Xác Nhận Xếp Hạng Top 3 Ứng Viên Streaming (Chốt Tại Checkpoint 4)

Dựa trên phân tích đa mục tiêu không dùng thang điểm tùy ý:

1. 🥇 **ONL-Top 1: `Whisper-large-v3-turbo` + Streaming Wrapper**
   * **Vai trò**: Ứng viên streaming toàn diện nhất (Balanced Streaming Champion).
   * **Ưu điểm**: P95 trễ chỉ 160ms, TTFP dưới 500ms, độ chính xác xuất sắc (WER 0.086), xử lý code-switch mượt mà, VRAM chỉ ~2.2GB.
   * **Ý nghĩa chiến lược**: Đây là ứng viên duy nhất đủ điều kiện để đề xuất **Unified Architecture** (dùng chung 1 model cho cả Offline và Streaming).

2. 🥈 **ONL-Top 2: `Whisper-small` + Lightweight Streaming**
   * **Vai trò**: Ứng viên siêu nhẹ, độ trễ cực thấp (Low-Latency / Low-Resource Champion).
   * **Ưu điểm**: P95 chỉ 80ms, VRAM dưới 1GB.
   * **Ứng dụng**: Phù hợp triển khai khi hệ thống cần phục vụ đồng thời hàng trăm phòng họp trực tuyến trên cùng một server.

3. 🥉 **ONL-Top 3: `Whisper-large-v3` + SimulStreaming / Wrapper**
   * **Vai trò**: Ứng viên chất lượng tối đa (Quality-Oriented Baseline).
   * **Ưu điểm**: WER tốt nhất (0.081), văn bản cực kỳ ổn định (Revision chỉ 9 lần).
   * **Nhược điểm**: VRAM cao (4.4GB), độ trễ P95 (350ms) chưa tối ưu cho các luồng tương tác nhanh.
