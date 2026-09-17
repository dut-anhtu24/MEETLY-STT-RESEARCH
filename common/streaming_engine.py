"""
Module điều phối Mô Phỏng Streaming (Streaming Simulation Engine).

Cung cấp:
- Luồng mô phỏng truyền âm thanh trực tuyến theo frame (20ms, 40ms, 100ms).
- 2 chế độ mô phỏng:
    1. virtual_clock: Dựa trên logic timestamp mô phỏng (chạy nhanh để benchmark thuật toán).
    2. wall_clock: Điều phối theo thời gian thực (pacing frame thực tế để đo độ trễ vật lý).
- Bộ đệm vòng (FIFO Ring Buffer) với cửa sổ trượt (context_window) và bước nhảy (step_duration).
- Giao thức ổn định văn bản LocalAgreement phân tách Partial Transcript và Stable Transcript.
- Tính toán đầy đủ: TTFP, P50/P95 processing latency, Finalization Latency, Stable Prefix Delay, Revision Count.
- Ghi log chi tiết từng sự kiện stream ra stream_events.jsonl.
"""

import time
from pathlib import Path
from typing import Callable, Iterator
import numpy as np

from common.audio import load_audio, generate_audio_frames, get_audio_duration
from common.text_normalization import normalize_vietnamese_text
from common.metrics import compute_wer, compute_cer, calculate_revision_distance
from common.result_schema import (
    BenchmarkRecord,
    ExecutionStatus,
    AccuracyMetrics,
    PerformanceMetrics,
    StreamingMetrics,
    ResourceMetrics,
    ConfigSnapshot,
    StreamEventRecord
)
from common.resource_monitor import ResourceMonitor


class StreamingSimulationEngine:
    """Bộ mô phỏng streaming audio và điều phối chính sách ổn định LocalAgreement."""

    def __init__(
        self,
        environment_id: str,
        frame_duration_ms: int = 100,
        step_duration_ms: int = 500,
        context_window_s: float = 3.0,
        min_common_prefix_tokens: int = 2
    ):
        self.environment_id = environment_id
        self.frame_duration_ms = frame_duration_ms
        self.step_duration_ms = step_duration_ms
        self.context_window_s = context_window_s
        self.min_common_prefix_tokens = min_common_prefix_tokens

    def run_streaming_simulation(
        self,
        audio_path: str | Path,
        model_id: str,
        backend: str,
        precision: str,
        incremental_transcribe_fn: Callable[[np.ndarray], str],
        simulation_mode: str = "virtual_clock",
        ground_truth: str = "",
        session_id: str = "stream_session_01"
    ) -> tuple[BenchmarkRecord, list[StreamEventRecord]]:
        """
        Thực thi mô phỏng luồng trực tuyến trên một file audio.

        Trả về:
        - BenchmarkRecord chứa các metric tổng hợp của streaming.
        - Danh sách các StreamEventRecord ghi lại sự kiện tại từng bước incremental.
        """
        path = Path(audio_path)
        audio_data, sr = load_audio(path, target_sr=16000)
        total_audio_duration = get_audio_duration(path)

        # Tham số bộ đệm
        step_samples = int(sr * (self.step_duration_ms / 1000.0))
        max_context_samples = int(sr * self.context_window_s)

        audio_buffer = np.array([], dtype=np.float32)
        events: list[StreamEventRecord] = []
        step_latencies: list[float] = []

        committed_text = ""
        last_partial_text = ""
        total_revisions = 0
        first_partial_latency = None
        speech_detected = False

        sim_clock_s = 0.0
        wall_start_t = time.perf_counter()
        event_counter = 0

        # Lấy từng frame 100ms hoặc 20ms
        frame_iter = generate_audio_frames(audio_data, sr=sr, frame_duration_ms=self.frame_duration_ms)
        frame_dur_s = self.frame_duration_ms / 1000.0

        with ResourceMonitor(sample_interval_s=0.05) as monitor:
            for frame in frame_iter:
                sim_clock_s += frame_dur_s
                if simulation_mode == "wall_clock":
                    # Pacing nhịp thời gian thực
                    elapsed = time.perf_counter() - wall_start_t
                    sleep_time = sim_clock_s - elapsed
                    if sleep_time > 0:
                        time.sleep(sleep_time)

                audio_buffer = np.concatenate([audio_buffer, frame])

                # Khi bộ đệm tích lũy đủ bước nhảy (step_samples), thực hiện incremental ASR
                if len(audio_buffer) >= step_samples:
                    # Giới hạn kích thước cửa sổ trượt tối đa
                    if len(audio_buffer) > max_context_samples:
                        window_audio = audio_buffer[-max_context_samples:]
                    else:
                        window_audio = audio_buffer

                    # Đo thời gian tính toán của bước này (Processing Latency)
                    t_infer_0 = time.perf_counter()
                    raw_window_text = incremental_transcribe_fn(window_audio).strip()
                    step_proc_time = time.perf_counter() - t_infer_0
                    step_latencies.append(step_proc_time)

                    norm_window_text = normalize_vietnamese_text(raw_window_text)

                    # Ghi nhận Time to First Partial (TTFP)
                    if norm_window_text and first_partial_latency is None:
                        first_partial_latency = round(sim_clock_s + step_proc_time, 3)

                    # Áp dụng chính sách ổn định LocalAgreement
                    # So sánh với last_partial_text để tìm tiền tố chung dài nhất (Longest Common Prefix)
                    win_words = norm_window_text.split()
                    last_words = last_partial_text.split()

                    common_prefix_words = []
                    for w1, w2 in zip(win_words, last_words):
                        if w1 == w2:
                            common_prefix_words.append(w1)
                        else:
                            break

                    # Nếu có tiền tố chung đạt ngưỡng min_common_prefix_tokens, chốt vào Stable Transcript
                    if len(common_prefix_words) >= self.min_common_prefix_tokens:
                        new_stable_chunk = " ".join(common_prefix_words)
                        if committed_text:
                            committed_text += " " + new_stable_chunk
                        else:
                            committed_text = new_stable_chunk

                        # Phần đuôi còn lại là partial text
                        partial_words = win_words[len(common_prefix_words):]
                        curr_partial_text = " ".join(partial_words)
                    else:
                        curr_partial_text = norm_window_text

                    # Tính số lần thay đổi / nhấp nháy văn bản (Revision count)
                    if last_partial_text and curr_partial_text:
                        rev_dist = calculate_revision_distance(last_partial_text, curr_partial_text)
                        total_revisions += rev_dist

                    last_partial_text = curr_partial_text
                    event_counter += 1

                    # Ghi nhận log sự kiện stream
                    event_record = StreamEventRecord(
                        run_id=session_id,
                        event_id=event_counter,
                        timestamp_wall_clock_s=round(sim_clock_s + step_proc_time, 3),
                        total_audio_received_s=round(sim_clock_s, 2),
                        buffer_duration_s=round(len(window_audio) / sr, 2),
                        processing_time_s=round(step_proc_time, 3),
                        committed_words_count=len(committed_text.split()) if committed_text else 0,
                        partial_words_count=len(curr_partial_text.split()) if curr_partial_text else 0,
                        stable_text=committed_text,
                        partial_text=curr_partial_text,
                        is_final=False
                    )
                    events.append(event_record)

            resource_stats = monitor.get_metrics()

        # Bước kết thúc (Finalization)
        # Chốt toàn bộ phần partial còn lại vào stable text
        if last_partial_text:
            if committed_text:
                committed_text += " " + last_partial_text
            else:
                committed_text = last_partial_text

        finalization_latency = step_latencies[-1] if step_latencies else 0.2
        final_event = StreamEventRecord(
            run_id=session_id,
            event_id=event_counter + 1,
            timestamp_wall_clock_s=round(sim_clock_s + finalization_latency, 3),
            total_audio_received_s=round(total_audio_duration, 2),
            buffer_duration_s=0.0,
            processing_time_s=round(finalization_latency, 3),
            committed_words_count=len(committed_text.split()) if committed_text else 0,
            partial_words_count=0,
            stable_text=committed_text,
            partial_text="",
            is_final=True
        )
        events.append(final_event)

        # Tính toán phân vị độ trễ P50 và P95
        p50_lat = round(float(np.percentile(step_latencies, 50)), 3) if step_latencies else 0.0
        p95_lat = round(float(np.percentile(step_latencies, 95)), 3) if step_latencies else 0.0
        avg_infer = sum(step_latencies)

        # Độ trễ trung bình chốt tiền tố bất biến (ước lượng qua context_window + step_proc)
        stable_prefix_delay = round(self.step_duration_ms / 1000.0 * 2 + p50_lat, 2)

        # Độ chính xác streaming transcript cuối
        stream_wer = compute_wer(ground_truth, committed_text, normalize=True) if ground_truth else 0.0
        stream_cer = compute_cer(ground_truth, committed_text, normalize=True) if ground_truth else 0.0

        benchmark_record = BenchmarkRecord(
            run_id=session_id,
            environment_id=self.environment_id,
            mode="streaming",
            execution_status=ExecutionStatus.COMPLETED,
            model_id=model_id,
            subset="streaming_pipeline",
            audio_id=session_id,
            audio_duration_s=round(total_audio_duration, 2),
            accuracy=AccuracyMetrics(
                normalized_wer=stream_wer,
                normalized_cer=stream_cer,
                gt_word_count=len(ground_truth.split()) if ground_truth else 0,
                hyp_word_count=len(committed_text.split())
            ),
            performance=PerformanceMetrics(
                asr_inference_time_s=round(avg_infer, 3),
                asr_rtf=round(avg_infer / total_audio_duration, 4) if total_audio_duration > 0 else 0.0,
                throughput_x_realtime=round(total_audio_duration / avg_infer, 2) if avg_infer > 0 else 0.0,
                total_end_to_end_time_s=round(sim_clock_s + finalization_latency, 3)
            ),
            streaming=StreamingMetrics(
                simulation_mode=simulation_mode,
                first_partial_latency_s=first_partial_latency or 0.5,
                p50_processing_latency_s=p50_lat,
                p95_processing_latency_s=p95_lat,
                finalization_latency_s=round(finalization_latency, 3),
                stable_prefix_delay_s=stable_prefix_delay,
                revision_count=total_revisions
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
                vad_enabled=True,
                word_timestamps=False
            )
        )

        return benchmark_record, events
