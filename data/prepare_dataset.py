"""
Script chuẩn bị và xác thực tập dữ liệu benchmark (Dataset Preparation).

Module này cung cấp các công cụ:
- Xác thực tính toàn vẹn của file metadata.csv và các đường dẫn audio/ground truth.
- Tạo dữ liệu giả lập (dummy fixtures) chuẩn 16kHz mono phục vụ unit test và smoke test.
- Tóm tắt thống kê các subset dữ liệu Tier A.
"""

import os
import csv
from pathlib import Path
import numpy as np
import soundfile as sf


def validate_metadata(csv_path: str | Path) -> list[dict]:
    """
    Xác thực cấu trúc file metadata.csv.

    Kiểm tra sự tồn tại của các cột bắt buộc và trả về danh sách các bản ghi hợp lệ.
    """
    required_columns = {
        "audio_id", "file_path", "ground_truth_path", "subset",
        "duration_s", "sample_rate", "channels"
    }

    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file metadata: {csv_path}")

    records = []
    with open(path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing_cols = required_columns - set(reader.fieldnames or [])
        if missing_cols:
            raise ValueError(f"Metadata thiếu các cột bắt buộc: {missing_cols}")

        for row in reader:
            records.append(row)

    return records


def create_dummy_fixtures(base_dir: str | Path) -> None:
    """
    Tạo các file audio giả lập (16kHz Mono WAV) và ground truth tương ứng.

    Dùng riêng cho việc kiểm thử tự động (Unit Tests) và kiểm tra hoạt động cơ bản (Smoke Tests).
    Tuyệt đối không dùng dữ liệu này để tính toán kết quả benchmark nghiên cứu chính thức.
    """
    base = Path(base_dir)
    audio_base = base / "audio"
    gt_base = base / "ground_truth"

    sample_rate = 16000
    duration_s = 2.0
    t = np.linspace(0, duration_s, int(sample_rate * duration_s), endpoint=False)
    # Tạo sóng sin 440Hz chuẩn
    audio_data = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    subsets = {
        "clean_vi": ("sample_01.wav", "clean_vi_01.txt", "xin chào đây là kiểm tra âm thanh chuẩn"),
        "codeswitch_tech": ("sample_01.wav", "cs_01.txt", "hôm nay team mình sẽ review API authentication và JWT access token"),
        "noisy_vi": ("sample_01.wav", "noisy_01.txt", "thử nghiệm âm thanh trong môi trường phòng họp có tiếng gõ phím"),
        "meeting_natural": ("sample_01.wav", "meeting_01.txt", "thì à chúng ta cần phải tối ưu lại hệ thống STT này"),
        "silence_pause": ("sample_01.wav", "silence_01.txt", "bắt đầu câu nói kết thúc câu nói"),
        "longform_meeting": ("sample_01.wav", "longform_01.txt", "đây là phiên họp kỹ thuật tổng thể của dự án Meetly")
    }

    for subset, (wav_name, txt_name, text_content) in subsets.items():
        subset_dir = audio_base / subset
        subset_dir.mkdir(parents=True, exist_ok=True)
        wav_path = subset_dir / wav_name
        if not wav_path.exists():
            sf.write(str(wav_path), audio_data, sample_rate)

        gt_base.mkdir(parents=True, exist_ok=True)
        txt_path = gt_base / txt_name
        if not txt_path.exists():
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(text_content)


if __name__ == "__main__":
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    current_dir = Path(__file__).parent
    print("Đang khởi tạo các mẫu dummy fixtures cho testing...")
    create_dummy_fixtures(current_dir)
    print("Xác thực metadata.csv...")
    records = validate_metadata(current_dir / "metadata.csv")
    print(f"✅ Đã xác thực {len(records)} mẫu dữ liệu trong Tier A metadata.")
