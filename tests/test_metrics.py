"""
Kiểm thử tự động cho module tính toán chỉ số đánh giá (common/metrics.py).
"""

import pytest
from common.metrics import (
    levenshtein_distance,
    compute_wer,
    compute_cer,
    calculate_rtf,
    calculate_throughput,
    calculate_revision_distance,
    calculate_technical_term_metrics
)


class TestMetrics:

    def test_levenshtein_distance_basic(self):
        """Kiểm tra khoảng cách Levenshtein cơ bản."""
        assert levenshtein_distance([], []) == 0
        assert levenshtein_distance(["a", "b"], ["a", "b"]) == 0
        assert levenshtein_distance(["a", "b"], ["a", "c"]) == 1  # Substitution
        assert levenshtein_distance(["a"], ["a", "b"]) == 1       # Insertion
        assert levenshtein_distance(["a", "b"], ["a"]) == 1       # Deletion

    def test_compute_wer(self):
        """Kiểm tra tính toán Word Error Rate (WER)."""
        ref = "chúng ta sẽ bắt đầu cuộc họp"
        # 1 thay thế ("hội nghị" thay cho "cuộc họp" = 1 thay thế / 1 xóa)
        hyp = "chúng ta sẽ bắt đầu hội nghị"
        # ref words = 7 words
        # hyp words = 6 words
        # substitutions/deletions: "cuộc", "họp" -> "hội", "nghị"
        wer = compute_wer(ref, hyp)
        assert 0.0 < wer <= 0.35

        # Hai câu giống hệt nhau
        assert compute_wer(ref, ref) == 0.0

        # Cả hai chuỗi rỗng
        assert compute_wer("", "") == 0.0

    def test_compute_cer(self):
        """Kiểm tra tính toán Character Error Rate (CER)."""
        ref = "gặp mặt"
        hyp = "gặp mat"  # sai dấu nặng ở chữ 'ặ' -> 'a'
        cer = compute_cer(ref, hyp)
        assert cer > 0.0
        assert cer < 0.25

    def test_rtf_and_throughput(self):
        """Kiểm tra tính toán RTF và Throughput."""
        audio_dur = 10.0
        infer_t = 2.0
        rtf = calculate_rtf(infer_t, audio_dur)
        throughput = calculate_throughput(infer_t, audio_dur)

        assert rtf == 0.2
        assert throughput == 5.0
        assert round(throughput * rtf, 2) == 1.0

    def test_revision_distance(self):
        """Kiểm tra khoảng cách thay đổi văn bản giữa 2 partials liên tiếp."""
        p1 = "chúng ta sẽ họp"
        p2 = "chúng ta sẽ học về API"
        # "họp" -> "học", thêm "về", thêm "API"
        dist = calculate_revision_distance(p1, p2)
        assert dist == 3

    def test_technical_term_metrics(self):
        """Kiểm tra hàm bao đo lường thuật ngữ kỹ thuật."""
        gt = "Hệ thống bảo mật dùng JWT access token và OAuth2"
        hyp = "Hệ thống bảo mật dùng JWT access token và OAuth2"
        terms = ["JWT", "OAuth2", "Spring Boot"]

        metrics = calculate_technical_term_metrics(gt, hyp, terms)
        assert metrics["technical_term_recall"] == 1.0
        assert metrics["technical_term_accuracy"] == 1.0
        assert metrics["term_correct"] == 2
