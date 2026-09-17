"""
Module xử lý và chuẩn hóa dữ liệu âm thanh (Audio Utilities).

Cung cấp các chức năng:
- Nạp file âm thanh, chuyển đổi về chuẩn 16 kHz Mono PCM float32.
- Lấy thông tin thời lượng âm thanh chính xác.
- Bộ sinh frame âm thanh phục vụ mô phỏng streaming (20ms, 40ms, 100ms).
- Trích xuất luồng âm thanh từ file video thông qua FFmpeg.
"""

import time
import subprocess
from pathlib import Path
from typing import Iterator
import numpy as np
import soundfile as sf
import scipy.signal


def load_audio(file_path: str | Path, target_sr: int = 16000) -> tuple[np.ndarray, int]:
    """
    Nạp file âm thanh và chuẩn hóa về 16kHz Mono float32 [-1.0, 1.0].

    Nếu file có nhiều kênh (stereo/multichannel), tự động lấy trung bình cộng các kênh.
    Nếu tần số lấy mẫu khác 16kHz, thực hiện resample chất lượng cao qua scipy.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file âm thanh: {file_path}")

    data, sr = sf.read(str(path), dtype="float32")

    # Chuyển đổi đa kênh sang Mono
    if data.ndim > 1:
        data = np.mean(data, axis=1)

    # Resample nếu tần số lấy mẫu khác mục tiêu
    if sr != target_sr:
        gcd = np.gcd(sr, target_sr)
        up = target_sr // gcd
        down = sr // gcd
        data = scipy.signal.resample_poly(data, up, down).astype(np.float32)
        sr = target_sr

    # Đảm bảo dải giá trị chuẩn hóa trong [-1.0, 1.0]
    max_val = np.max(np.abs(data))
    if max_val > 1.0:
        data = data / max_val

    return data, sr


def get_audio_duration(file_path: str | Path) -> float:
    """
    Lấy thời lượng âm thanh (tính bằng giây) từ file mà không cần load toàn bộ dữ liệu vào RAM.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {file_path}")

    info = sf.info(str(path))
    return float(info.duration)


def save_audio(file_path: str | Path, audio: np.ndarray, sr: int = 16000) -> None:
    """
    Lưu mảng âm thanh numpy thành file WAV chuẩn 16-bit PCM hoặc float32.
    """
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, sr)


def generate_audio_frames(
    audio: np.ndarray,
    sr: int = 16000,
    frame_duration_ms: int = 100
) -> Iterator[np.ndarray]:
    """
    Chia mảng âm thanh thành các frame nhỏ phục vụ mô phỏng streaming.

    Ví dụ: Với sr=16000 và frame_duration_ms=100, mỗi frame chứa đúng 1600 mẫu.
    """
    frame_size = int(sr * (frame_duration_ms / 1000.0))
    total_samples = len(audio)

    for start_idx in range(0, total_samples, frame_size):
        end_idx = min(start_idx + frame_size, total_samples)
        frame = audio[start_idx:end_idx]
        # Nếu frame cuối ngắn hơn frame_size, đệm thêm số 0 để giữ kích thước cố định
        if len(frame) < frame_size:
            padded = np.zeros(frame_size, dtype=np.float32)
            padded[:len(frame)] = frame
            yield padded
        else:
            yield frame.astype(np.float32)


def extract_audio_from_video(
    video_path: str | Path,
    output_wav_path: str | Path,
    target_sr: int = 16000
) -> float:
    """
    Trích xuất âm thanh từ file video bằng FFmpeg, chuyển thẳng về 16kHz Mono WAV.

    Trả về: Thời gian thực hiện lệnh trích xuất (tính bằng giây).
    """
    v_path = Path(video_path)
    out_path = Path(output_wav_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not v_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file video: {video_path}")

    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-i", str(v_path),
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", str(target_sr),
        "-ac", "1",
        str(out_path)
    ]

    t0 = time.perf_counter()
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except (subprocess.SubprocessError, FileNotFoundError) as e:
        # Nếu máy chưa cài ffmpeg (ví dụ local test), ghi nhận fallback
        raise RuntimeError(f"Lỗi khi thực thi FFmpeg để trích xuất video: {e}")

    extraction_time = time.perf_counter() - t0
    return extraction_time
