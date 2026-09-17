"""
Kiểm thử tự động cho module xử lý âm thanh (common/audio.py).
"""

import tempfile
from pathlib import Path
import numpy as np
import pytest
from common.audio import (
    load_audio,
    save_audio,
    get_audio_duration,
    generate_audio_frames
)


class TestAudio:

    def test_save_and_load_audio_mono(self):
        """Kiểm tra lưu và nạp âm thanh mono 16kHz chuẩn hóa [-1.0, 1.0]."""
        sr = 16000
        duration_s = 1.0
        t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)
        orig_data = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)

        with tempfile.TemporaryDirectory() as tmp_dir:
            wav_path = Path(tmp_dir) / "test_tone.wav"
            save_audio(wav_path, orig_data, sr=sr)

            assert wav_path.exists()

            # Nạp lại và kiểm tra
            loaded_data, loaded_sr = load_audio(wav_path, target_sr=sr)
            assert loaded_sr == sr
            assert loaded_data.ndim == 1
            assert len(loaded_data) == int(sr * duration_s)
            assert np.max(np.abs(loaded_data)) <= 1.0

            # Kiểm tra lấy thời lượng
            dur = get_audio_duration(wav_path)
            assert pytest.approx(dur, 0.05) == duration_s

    def test_generate_audio_frames(self):
        """Kiểm tra bộ sinh frame streaming (100ms và 20ms)."""
        sr = 16000
        duration_s = 1.0  # 16000 mẫu
        dummy_audio = np.ones(int(sr * duration_s), dtype=np.float32)

        # Test frame 100ms -> 1600 samples/frame -> 10 frames
        frames_100ms = list(generate_audio_frames(dummy_audio, sr=sr, frame_duration_ms=100))
        assert len(frames_100ms) == 10
        assert all(len(f) == 1600 for f in frames_100ms)

        # Test frame 20ms -> 320 samples/frame -> 50 frames
        frames_20ms = list(generate_audio_frames(dummy_audio, sr=sr, frame_duration_ms=20))
        assert len(frames_20ms) == 50
        assert all(len(f) == 320 for f in frames_20ms)
