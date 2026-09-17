"""
Kiểm thử tự động cho module chuẩn hóa văn bản tiếng Việt (common/text_normalization.py).
"""

import unicodedata
import pytest
from common.text_normalization import normalize_vietnamese_text, analyze_technical_terms


class TestTextNormalization:

    def test_unicode_nfc_consistency(self):
        """Kiểm tra việc đồng nhất biểu diễn Unicode giữa kiểu gõ cũ và mới (NFD vs NFC)."""
        # "hoà" gõ tổ hợp (NFD) vs "hòa" dựng sẵn (NFC)
        text_nfd = unicodedata.normalize("NFD", "hòa bình và tự do")
        text_nfc = unicodedata.normalize("NFC", "hoà bình và tự do")

        norm1 = normalize_vietnamese_text(text_nfd)
        norm2 = normalize_vietnamese_text(text_nfc)

        assert norm1 == norm2
        assert norm1 == "hòa bình và tự do"

    def test_punctuation_and_whitespace_removal(self):
        """Kiểm tra loại bỏ dấu câu, dấu ngoặc kép và chuẩn hóa khoảng trắng thừa."""
        raw_text = '  "Xin chào,   Meetly-STT!": hôm nay... ngày 17/09/2026?  '
        normalized = normalize_vietnamese_text(raw_text)
        expected = "xin chào meetly stt hôm nay ngày 17 09 2026"
        assert normalized == expected

    def test_technical_term_analysis_exact(self):
        """Kiểm tra phân tích thuật ngữ kỹ thuật khi mô hình nhận diện chính xác 100%."""
        gt = "Chúng ta sẽ dùng API authentication và JWT access token trên Spring Boot."
        hyp = "Chúng ta sẽ dùng API authentication và JWT access token trên Spring Boot."
        terms = ["API", "JWT", "Spring Boot", "Docker"]

        result = analyze_technical_terms(gt, hyp, terms)
        assert result["total_in_gt"] == 3  # Docker không có trong GT
        assert result["correct"] == 3
        assert result["substitution"] == 0
        assert result["deletion"] == 0
        assert result["term_recall"] == 1.0
        assert result["term_accuracy"] == 1.0

    def test_technical_term_analysis_with_errors(self):
        """Kiểm tra phân tích thuật ngữ khi có lỗi substitution và deletion."""
        gt = "Hệ thống backend sử dụng Docker và Kubernetes để deploy microservice."
        # Giả lập model nhận dạng sai Docker thành "đốc cơ" (deletion) và Kubernetes thành "Kuber" (substitution)
        hyp = "Hệ thống backend sử dụng đốc cơ và Kuber để deploy microservice."
        terms = ["backend", "Docker", "Kubernetes", "microservice"]

        result = analyze_technical_terms(gt, hyp, terms)
        assert result["total_in_gt"] == 4
        assert result["correct"] == 2  # backend, microservice
        assert result["substitution"] == 1  # Kubernetes -> Kuber
        assert result["deletion"] == 1  # Docker -> hoàn toàn mất
        assert result["term_recall"] == 0.5
