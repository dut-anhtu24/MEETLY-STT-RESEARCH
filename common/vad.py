"""
Module Voice Activity Detection (Silero VAD Wrapper).

Cung cấp:
- Lớp SileroVADWrapper tải mô hình Silero VAD qua ONNX hoặc PyTorch JIT.
- Phân đoạn tiếng nói (speech timestamps) phục vụ cắt audio thành các segment thoại.
- Cơ chế fallback dự phòng cho môi trường offline hoặc unit test nội bộ.
"""

from typing import TypedDict
import numpy as np
import torch


class SpeechSegment(TypedDict):
    start_s: float
    end_s: float
    start_sample: int
    end_sample: int


class SileroVADWrapper:
    """Wrapper bao bọc mô hình Silero VAD với cơ chế xử lý lỗi và fallback."""

    def __init__(self, onnx: bool = False, device: str = "cpu"):
        self.device = device
        self.onnx = onnx
        self.model = None
        self._utils = None
        self.is_loaded = False
        self._load_model()

    def _load_model(self) -> None:
        """Nạp model Silero VAD từ Torch Hub với fallback an toàn."""
        try:
            model, utils = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=False,
                onnx=self.onnx,
                trust_repo=True
            )
            self.model = model
            self._utils = utils
            self.is_loaded = True
        except Exception as e:
            # Fallback nếu máy không có mạng hoặc chưa tải được torch hub
            self.is_loaded = False
            self.model = None

    def get_speech_timestamps(
        self,
        audio: np.ndarray,
        sampling_rate: int = 16000,
        threshold: float = 0.5,
        min_speech_duration_ms: int = 250,
        min_silence_duration_ms: int = 100
    ) -> list[SpeechSegment]:
        """
        Xác định các khoảng thời gian có giọng nói trong mảng âm thanh.

        Trả về danh sách các phân đoạn gồm thời gian (giây) và chỉ số mẫu (samples).
        """
        if self.is_loaded and self.model is not None and self._utils is not None:
            get_timestamps, _, _, _, _ = self._utils
            tensor_audio = torch.from_numpy(audio).float()
            if self.device != "cpu" and torch.cuda.is_available():
                tensor_audio = tensor_audio.to(self.device)

            raw_segments = get_timestamps(
                tensor_audio,
                self.model,
                threshold=threshold,
                sampling_rate=sampling_rate,
                min_speech_duration_ms=min_speech_duration_ms,
                min_silence_duration_ms=min_silence_duration_ms
            )

            result: list[SpeechSegment] = []
            for seg in raw_segments:
                start_sample = int(seg["start"])
                end_sample = int(seg["end"])
                result.append({
                    "start_s": round(start_sample / sampling_rate, 3),
                    "end_s": round(end_sample / sampling_rate, 3),
                    "start_sample": start_sample,
                    "end_sample": end_sample
                })
            return result

        # Fallback dựa trên năng lượng tín hiệu (Energy-based VAD) cho unit test / offline
        return self._energy_vad_fallback(audio, sampling_rate, threshold)

    def _energy_vad_fallback(
        self,
        audio: np.ndarray,
        sampling_rate: int,
        threshold: float
    ) -> list[SpeechSegment]:
        """
        Fallback đơn giản tính toán ngưỡng năng lượng RMS cho các frame 50ms khi model chưa sẵn sàng.
        """
        frame_size = int(sampling_rate * 0.05)
        if len(audio) < frame_size:
            return [{
                "start_s": 0.0,
                "end_s": round(len(audio) / sampling_rate, 3),
                "start_sample": 0,
                "end_sample": len(audio)
            }]

        segments: list[SpeechSegment] = []
        is_speech = False
        start_sample = 0

        # Ngưỡng năng lượng RMS tương đối
        rms_threshold = max(0.01, threshold * 0.05)

        for i in range(0, len(audio), frame_size):
            frame = audio[i:i + frame_size]
            rms = np.sqrt(np.mean(frame**2)) if len(frame) > 0 else 0.0

            if rms >= rms_threshold and not is_speech:
                is_speech = True
                start_sample = i
            elif rms < rms_threshold and is_speech:
                is_speech = False
                end_sample = i
                segments.append({
                    "start_s": round(start_sample / sampling_rate, 3),
                    "end_s": round(end_sample / sampling_rate, 3),
                    "start_sample": start_sample,
                    "end_sample": end_sample
                })

        if is_speech:
            segments.append({
                "start_s": round(start_sample / sampling_rate, 3),
                "end_s": round(len(audio) / sampling_rate, 3),
                "start_sample": start_sample,
                "end_sample": len(audio)
            })

        # Nếu không có đoạn nào vượt ngưỡng, trả về toàn bộ audio như 1 đoạn
        if not segments:
            segments.append({
                "start_s": 0.0,
                "end_s": round(len(audio) / sampling_rate, 3),
                "start_sample": 0,
                "end_sample": len(audio)
            })

        return segments
