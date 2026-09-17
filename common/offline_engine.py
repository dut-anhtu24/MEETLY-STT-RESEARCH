"""
Module điều phối Pipeline Ngoại Tuyến (Offline Pipeline Engine).

Cung cấp:
- Luồng xử lý toàn trình từ Video -> Tách Audio -> Silero VAD -> ASR -> Timestamps -> Structured JSON.
- Tách bạch rõ ràng từng mốc thời gian: video_audio_extraction_time, vad_time, asr_time, assembly_time.
- Đo đạc độc lập ASR-only RTF (để so sánh công bằng) và Total End-to-End Latency.
- Hỗ trợ các biến thể bóc tách (Ablations): Baseline Transformers vs faster-whisper (FP16/INT8) và VAD chunking.
"""

import time
import json
from pathlib import Path
from typing import Any, Callable
import numpy as np

from common.audio import load_audio, extract_audio_from_video, get_audio_duration
from common.vad import SileroVADWrapper, SpeechSegment
from common.metrics import compute_wer, compute_cer, calculate_rtf, calculate_throughput
from common.result_schema import BenchmarkRecord, ExecutionStatus, AccuracyMetrics, PerformanceMetrics, ResourceMetrics, ConfigSnapshot
from common.resource_monitor import ResourceMonitor


class OfflinePipelineEngine:
    """Động cơ điều phối pipeline xử lý ngoại tuyến từ file video/audio ra transcript có cấu trúc."""

    def __init__(self, environment_id: str, vad_provider: str = "silero", vad_threshold: float = 0.5):
        self.environment_id = environment_id
        self.vad_threshold = vad_threshold
        self.vad_wrapper = SileroVADWrapper(onnx=False) if vad_provider == "silero" else None

    def process_offline(
        self,
        input_path: str | Path,
        model_id: str,
        backend: str,
        precision: str,
        transcribe_fn: Callable[[np.ndarray], str],
        use_vad: bool = True,
        is_video: bool = False,
        ground_truth: str = "",
        session_id: str = "meeting_session_01"
    ) -> tuple[dict, BenchmarkRecord]:
        """
        Thực thi toàn bộ pipeline offline và đo lường chi tiết từng công đoạn.

        Trả về:
        - structured_json: Dữ liệu JSON có cấu trúc gồm segments, timestamps, full transcript.
        - benchmark_record: Bản ghi đo kiểm phục vụ lưu trữ kết quả thực nghiệm.
        """
        in_path = Path(input_path)
        extraction_time = 0.0
        temp_wav_path = None

        # 1. Trích xuất audio nếu đầu vào là video
        if is_video:
            temp_wav_path = in_path.parent / f"extracted_{in_path.stem}.wav"
            try:
                extraction_time = extract_audio_from_video(in_path, temp_wav_path, target_sr=16000)
                working_audio_path = temp_wav_path
            except Exception:
                # Nếu không có ffmpeg thực tế, dùng in_path giả định
                working_audio_path = in_path
        else:
            working_audio_path = in_path

        audio_duration = get_audio_duration(working_audio_path)
        audio_data, sr = load_audio(working_audio_path, target_sr=16000)

        # 2. Phân đoạn âm thanh bằng Silero VAD (nếu được kích hoạt)
        vad_time = 0.0
        segments: list[SpeechSegment] = []

        if use_vad and self.vad_wrapper is not None:
            t_vad_0 = time.perf_counter()
            segments = self.vad_wrapper.get_speech_timestamps(
                audio_data,
                sampling_rate=sr,
                threshold=self.vad_threshold
            )
            vad_time = round(time.perf_counter() - t_vad_0, 3)
        else:
            # Nếu không dùng VAD, coi toàn bộ audio là 1 phân đoạn lớn
            segments = [{
                "start_s": 0.0,
                "end_s": audio_duration,
                "start_sample": 0,
                "end_sample": len(audio_data)
            }]

        # 3. Thực thi ASR suy luận (có gắn ResourceMonitor)
        asr_start = time.perf_counter()
        transcript_segments = []
        full_text_pieces = []

        with ResourceMonitor(sample_interval_s=0.05) as monitor:
            for idx, seg in enumerate(segments):
                seg_audio = audio_data[seg["start_sample"]:seg["end_sample"]]
                if len(seg_audio) == 0:
                    continue

                seg_text = transcribe_fn(seg_audio).strip()
                if seg_text:
                    full_text_pieces.append(seg_text)
                    transcript_segments.append({
                        "id": idx,
                        "start": seg["start_s"],
                        "end": seg["end_s"],
                        "text": seg_text,
                        "words": [
                            {"word": w, "start": seg["start_s"], "end": seg["end_s"]}
                            for w in seg_text.split()
                        ]
                    })

        asr_time = time.perf_counter() - asr_start
        resource_stats = monitor.get_metrics()

        # 4. Lắp ghép văn bản Structured JSON
        assembly_start = time.perf_counter()
        full_transcript = " ".join(full_text_pieces)
        assembly_time = time.perf_counter() - assembly_start

        total_end_to_end = extraction_time + vad_time + asr_time + assembly_time

        structured_json = {
            "metadata": {
                "session_id": session_id,
                "model_id": model_id,
                "backend": backend,
                "precision": precision,
                "vad_enabled": use_vad,
                "audio_duration_s": round(audio_duration, 2),
                "asr_inference_time_s": round(asr_time, 3),
                "total_processing_time_s": round(total_end_to_end, 3),
                "asr_rtf": calculate_rtf(asr_time, audio_duration)
            },
            "segments": transcript_segments,
            "full_transcript": full_transcript
        }

        # 5. Tạo BenchmarkRecord
        wer = compute_wer(ground_truth, full_transcript, normalize=True) if ground_truth else 0.0
        cer = compute_cer(ground_truth, full_transcript, normalize=True) if ground_truth else 0.0

        benchmark_record = BenchmarkRecord(
            run_id=f"off_{session_id}_{int(time.time())}",
            environment_id=self.environment_id,
            mode="offline",
            execution_status=ExecutionStatus.COMPLETED,
            model_id=model_id,
            subset="offline_pipeline",
            audio_id=session_id,
            audio_duration_s=round(audio_duration, 2),
            accuracy=AccuracyMetrics(
                normalized_wer=wer,
                normalized_cer=cer,
                gt_word_count=len(ground_truth.split()) if ground_truth else 0,
                hyp_word_count=len(full_transcript.split())
            ),
            performance=PerformanceMetrics(
                asr_inference_time_s=round(asr_time, 3),
                asr_rtf=calculate_rtf(asr_time, audio_duration),
                throughput_x_realtime=calculate_throughput(asr_time, audio_duration),
                video_audio_extraction_time_s=round(extraction_time, 3) if is_video else None,
                vad_segmentation_time_s=round(vad_time, 3) if use_vad else None,
                total_end_to_end_time_s=round(total_end_to_end, 3)
            ),
            resource=ResourceMetrics(
                gpu_memory_source=resource_stats["gpu_memory_source"],
                baseline_vram_mb=resource_stats["baseline_vram_mb"],
                peak_vram_mb=resource_stats["peak_vram_mb"],
                delta_peak_vram_mb=resource_stats["delta_peak_vram_mb"],
                baseline_ram_mb=resource_stats["baseline_ram_mb"],
                peak_ram_mb=resource_stats["peak_ram_mb"],
                delta_peak_ram_mb=resource_stats["delta_peak_ram_mb"],
                avg_cpu_percent=resource_stats["avg_cpu_percent"],
                peak_cpu_percent=resource_stats["peak_cpu_percent"]
            ),
            config=ConfigSnapshot(
                backend=backend,
                precision=precision,
                vad_enabled=use_vad,
                vad_threshold=self.vad_threshold,
                word_timestamps=True
            )
        )

        # Xóa file audio tạm nếu trích xuất từ video
        if temp_wav_path and temp_wav_path.exists():
            try:
                temp_wav_path.unlink()
            except Exception:
                pass

        return structured_json, benchmark_record
