"""Tests for text normalization."""

import pytest

from orontts.preprocessing.text import (
    MongolianTextNormalizer,
    number_to_mongolian,
    normalize_text,
)


class TestNumberToMongolian:
    """Tests for number-to-text conversion."""

    def test_zero(self) -> None:
        assert number_to_mongolian(0) == "тэг"

    def test_single_digits(self) -> None:
        assert number_to_mongolian(1) == "нэг"
        assert number_to_mongolian(5) == "тав"
        assert number_to_mongolian(9) == "ес"

    def test_teens(self) -> None:
        assert number_to_mongolian(10) == "арав"
        assert number_to_mongolian(11) == "арван нэг"
        assert number_to_mongolian(15) == "арван тав"
        assert number_to_mongolian(19) == "арван ес"

    def test_tens(self) -> None:
        assert number_to_mongolian(20) == "хорь"
        assert number_to_mongolian(21) == "хорин нэг"
        assert number_to_mongolian(30) == "гуч"
        assert number_to_mongolian(55) == "тавин тав"
        assert number_to_mongolian(99) == "ерэн ес"

    def test_hundreds(self) -> None:
        assert number_to_mongolian(100) == "зуу"
        assert number_to_mongolian(101) == "зуун нэг"
        assert number_to_mongolian(200) == "хоёр зуу"
        assert number_to_mongolian(500) == "тав зуу"

    def test_thousands(self) -> None:
        assert "мянга" in number_to_mongolian(1000)
        assert "мянга" in number_to_mongolian(2000)

    def test_negative(self) -> None:
        result = number_to_mongolian(-5)
        assert "хасах" in result
        assert "тав" in result

    def test_float(self) -> None:
        result = number_to_mongolian(3.14)
        assert "гурав" in result
        assert "бүтэн" in result

    def test_string_input(self) -> None:
        assert number_to_mongolian("42") == number_to_mongolian(42)

    def test_ordinal(self) -> None:
        result = number_to_mongolian(1, ordinal=True)
        assert "нэгд" in result or "дугаар" in result


class TestMongolianTextNormalizer:
    """Tests for text normalizer."""

    def test_basic_normalization(self) -> None:
        normalizer = MongolianTextNormalizer()
        text = "Сайн байна уу"
        result = normalizer.normalize(text)
        assert result == "Сайн байна уу"

    def test_number_conversion(self) -> None:
        normalizer = MongolianTextNormalizer()
        text = "Энэ 3 ширээ байна"
        result = normalizer.normalize(text)
        assert "гурав" in result
        assert "3" not in result

    def test_abbreviation_expansion(self) -> None:
        normalizer = MongolianTextNormalizer()
        text = "5 км зам"
        result = normalizer.normalize(text)
        assert "километр" in result

    def test_currency(self) -> None:
        normalizer = MongolianTextNormalizer()
        text = "1000₮"
        result = normalizer.normalize(text)
        assert "төгрөг" in result

    def test_percentage(self) -> None:
        normalizer = MongolianTextNormalizer()
        text = "50%"
        result = normalizer.normalize(text)
        assert "хувь" in result

    def test_whitespace_cleanup(self) -> None:
        normalizer = MongolianTextNormalizer()
        text = "Сайн    байна   уу"
        result = normalizer.normalize(text)
        assert "    " not in result
        assert result == "Сайн байна уу"

    def test_no_number_conversion(self) -> None:
        from orontts.preprocessing.text import MongolianTextNormalizerConfig

        config = MongolianTextNormalizerConfig(convert_numbers=False)
        normalizer = MongolianTextNormalizer(config)
        text = "5 ширээ"
        result = normalizer.normalize(text)
        assert "5" in result


class TestNormalizeTextFunction:
    """Tests for convenience function."""

    def test_normalize_text(self) -> None:
        result = normalize_text("Сайн байна уу")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_normalize_text_with_options(self) -> None:
        result = normalize_text("5 км", expand_abbreviations=True)
        assert "километр" in result
