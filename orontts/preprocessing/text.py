"""Mongolian Cyrillic text normalization with number-to-text conversion."""

import re
from dataclasses import dataclass, field
from typing import Self

from orontts.exceptions import TextNormalizationError


# Mongolian number words (cardinal)
ONES = {
    0: "",
    1: "нэг",
    2: "хоёр",
    3: "гурав",
    4: "дөрөв",
    5: "тав",
    6: "зургаа",
    7: "долоо",
    8: "найм",
    9: "ес",
}

TEENS = {
    10: "арав",
    11: "арван нэг",
    12: "арван хоёр",
    13: "арван гурав",
    14: "арван дөрөв",
    15: "арван тав",
    16: "арван зургаа",
    17: "арван долоо",
    18: "арван найм",
    19: "арван ес",
}

TENS = {
    2: "хорь",
    3: "гуч",
    4: "дөч",
    5: "тавь",
    6: "жар",
    7: "дал",
    8: "ная",
    9: "ер",
}

# Connector forms for tens (when followed by ones)
TENS_CONNECTOR = {
    2: "хорин",
    3: "гучин",
    4: "дөчин",
    5: "тавин",
    6: "жаран",
    7: "далан",
    8: "наян",
    9: "ерэн",
}

HUNDREDS = "зуу"
HUNDREDS_CONNECTOR = "зуун"

THOUSANDS = "мянга"
THOUSANDS_CONNECTOR = "мянган"

MILLIONS = "сая"
BILLIONS = "тэрбум"
TRILLIONS = "их наяд"

# Ordinal suffixes (simplified - depends on vowel harmony)
ORDINAL_SUFFIXES = {
    "hard": "дугаар",  # After back vowels (а, о, у)
    "soft": "дүгээр",  # After front vowels (э, ө, ү)
    "neutral": "дугаар",  # Default
}

# Common abbreviations
ABBREVIATIONS = {
    "др": "доктор",
    "проф": "профессор",
    "ш.м": "шинэ монгол",
    "т.": "товчлол",
    "хн": "хүн",
    "жил": "жил",
    "сар": "сар",
    "өдөр": "өдөр",
    "цаг": "цаг",
    "мин": "минут",
    "сек": "секунд",
    "км": "километр",
    "м": "метр",
    "см": "сантиметр",
    "мм": "миллиметр",
    "кг": "килограмм",
    "г": "грамм",
    "л": "литр",
    "мл": "миллилитр",
}


@dataclass
class MongolianTextNormalizerConfig:
    """Configuration for text normalization."""

    expand_abbreviations: bool = True
    convert_numbers: bool = True
    normalize_punctuation: bool = True
    lowercase: bool = False
    remove_extra_spaces: bool = True
    custom_replacements: dict[str, str] = field(default_factory=dict)


def _get_vowel_class(word: str) -> str:
    """Determine vowel harmony class of a word."""
    back_vowels = set("аоуы")
    front_vowels = set("эөүий")

    for char in reversed(word.lower()):
        if char in back_vowels:
            return "hard"
        if char in front_vowels:
            return "soft"
    return "neutral"


def _convert_two_digits(num: int) -> str:
    """Convert a two-digit number (0-99) to Mongolian words."""
    if num == 0:
        return ""
    if num < 10:
        return ONES[num]
    if num < 20:
        return TEENS[num]

    tens, ones = divmod(num, 10)
    if ones == 0:
        return TENS[tens]
    return f"{TENS_CONNECTOR[tens]} {ONES[ones]}"


def _convert_three_digits(num: int) -> str:
    """Convert a three-digit number (0-999) to Mongolian words."""
    if num == 0:
        return ""
    if num < 100:
        return _convert_two_digits(num)

    hundreds, remainder = divmod(num, 100)
    hundreds_word = ONES[hundreds] if hundreds > 1 else ""

    if remainder == 0:
        return f"{hundreds_word} {HUNDREDS}".strip()

    return f"{hundreds_word} {HUNDREDS_CONNECTOR} {_convert_two_digits(remainder)}".strip()


def number_to_mongolian(
    num: int | float | str,
    ordinal: bool = False,
) -> str:
    """Convert a number to Mongolian words.

    Handles cardinal and ordinal numbers with proper grammatical forms.

    Args:
        num: Number to convert (int, float, or numeric string).
        ordinal: If True, produce ordinal form (e.g., "нэгдүгээр").

    Returns:
        Mongolian word representation of the number.

    Raises:
        TextNormalizationError: If conversion fails.

    Examples:
        >>> number_to_mongolian(21)
        'хорин нэг'
        >>> number_to_mongolian(100)
        'зуу'
        >>> number_to_mongolian(1, ordinal=True)
        'нэгдүгээр'
    """
    try:
        # Handle string input
        if isinstance(num, str):
            num = num.replace(",", "").replace(" ", "")
            if "." in num:
                num = float(num)
            else:
                num = int(num)

        # Handle floats
        if isinstance(num, float):
            int_part = int(num)
            dec_part = str(num).split(".")[1] if "." in str(num) else ""
            result = number_to_mongolian(int_part)
            if dec_part:
                dec_words = " ".join(
                    number_to_mongolian(int(d)) if d != "0" else "тэг"
                    for d in dec_part
                )
                result = f"{result} бүтэн {dec_words}"
            return result

        # Handle negative numbers
        if num < 0:
            return f"хасах {number_to_mongolian(abs(num))}"

        # Zero
        if num == 0:
            return "тэг"

        # Build number word
        parts: list[str] = []

        # Trillions (10^12)
        if num >= 1_000_000_000_000:
            trillions, num = divmod(num, 1_000_000_000_000)
            parts.append(f"{_convert_three_digits(trillions)} {TRILLIONS}")

        # Billions (10^9)
        if num >= 1_000_000_000:
            billions, num = divmod(num, 1_000_000_000)
            parts.append(f"{_convert_three_digits(billions)} {BILLIONS}")

        # Millions (10^6)
        if num >= 1_000_000:
            millions, num = divmod(num, 1_000_000)
            parts.append(f"{_convert_three_digits(millions)} {MILLIONS}")

        # Thousands (10^3)
        if num >= 1000:
            thousands, num = divmod(num, 1000)
            th_word = _convert_three_digits(thousands)
            if num > 0:
                parts.append(f"{th_word} {THOUSANDS_CONNECTOR}")
            else:
                parts.append(f"{th_word} {THOUSANDS}")

        # Remainder (0-999)
        if num > 0:
            parts.append(_convert_three_digits(num))

        result = " ".join(parts).strip()

        # Convert to ordinal if requested
        if ordinal:
            vowel_class = _get_vowel_class(result)
            suffix = ORDINAL_SUFFIXES[vowel_class]
            # Special cases for ordinals
            if result.endswith("нэг"):
                result = result[:-3] + "нэгд" + ("үгээр" if vowel_class == "soft" else "угаар")
            elif result.endswith("хоёр"):
                result = result[:-4] + "хоёрд" + ("угаар" if vowel_class == "hard" else "үгээр")
            else:
                result = result + suffix

        return result

    except Exception as e:
        raise TextNormalizationError(f"Failed to convert number {num}: {e}") from e


class MongolianTextNormalizer:
    """Text normalizer for Mongolian Cyrillic text.

    Handles number expansion, abbreviations, and text cleaning.
    """

    def __init__(self, config: MongolianTextNormalizerConfig | None = None) -> None:
        """Initialize normalizer.

        Args:
            config: Normalization configuration.
        """
        self.config = config or MongolianTextNormalizerConfig()

        # Compile regex patterns
        self._number_pattern = re.compile(r"\b\d+([.,]\d+)?\b")
        self._ordinal_pattern = re.compile(r"\b(\d+)[-.]?(дугаар|дүгээр|р|д)\b", re.IGNORECASE)
        self._phone_pattern = re.compile(r"\b\d{4}[-\s]?\d{4}\b")
        self._year_pattern = re.compile(r"\b(19|20)\d{2}\s*он\b")
        self._time_pattern = re.compile(r"\b(\d{1,2}):(\d{2})\b")
        self._date_pattern = re.compile(r"\b(\d{1,2})[./](\d{1,2})[./](\d{2,4})\b")
        self._currency_pattern = re.compile(r"(\d+(?:[.,]\d+)?)\s*₮")
        self._percent_pattern = re.compile(r"(\d+(?:[.,]\d+)?)\s*%")
        self._whitespace_pattern = re.compile(r"\s+")

    def normalize(self, text: str) -> str:
        """Normalize Mongolian text.

        Args:
            text: Input text to normalize.

        Returns:
            Normalized text.

        Raises:
            TextNormalizationError: If normalization fails.
        """
        try:
            # Apply custom replacements first
            for pattern, replacement in self.config.custom_replacements.items():
                text = text.replace(pattern, replacement)

            # Expand abbreviations
            if self.config.expand_abbreviations:
                text = self._expand_abbreviations(text)

            # Convert numbers
            if self.config.convert_numbers:
                text = self._convert_all_numbers(text)

            # Normalize punctuation
            if self.config.normalize_punctuation:
                text = self._normalize_punctuation(text)

            # Case normalization
            if self.config.lowercase:
                text = text.lower()

            # Clean up whitespace
            if self.config.remove_extra_spaces:
                text = self._whitespace_pattern.sub(" ", text).strip()

            return text

        except Exception as e:
            raise TextNormalizationError(f"Normalization failed: {e}") from e

    def _expand_abbreviations(self, text: str) -> str:
        """Expand known abbreviations."""
        for abbrev, expansion in ABBREVIATIONS.items():
            # Match abbreviation with word boundaries
            pattern = rf"\b{re.escape(abbrev)}\.?\b"
            text = re.sub(pattern, expansion, text, flags=re.IGNORECASE)
        return text

    def _convert_all_numbers(self, text: str) -> str:
        """Convert all numeric patterns to words."""
        # Handle special patterns first (order matters)

        # Currency
        text = self._currency_pattern.sub(
            lambda m: f"{number_to_mongolian(m.group(1))} төгрөг", text
        )

        # Percentages
        text = self._percent_pattern.sub(
            lambda m: f"{number_to_mongolian(m.group(1))} хувь", text
        )

        # Time (HH:MM)
        text = self._time_pattern.sub(
            lambda m: f"{number_to_mongolian(int(m.group(1)))} цаг "
            f"{number_to_mongolian(int(m.group(2)))} минут",
            text,
        )

        # Ordinals
        text = self._ordinal_pattern.sub(
            lambda m: number_to_mongolian(int(m.group(1)), ordinal=True), text
        )

        # Plain numbers
        text = self._number_pattern.sub(
            lambda m: number_to_mongolian(m.group(0)), text
        )

        return text

    def _normalize_punctuation(self, text: str) -> str:
        """Normalize punctuation marks."""
        # Standardize quotes
        text = re.sub(r'[""«»]', '"', text)
        text = re.sub(r"['']", "'", text)

        # Standardize dashes
        text = re.sub(r"[–—]", "-", text)

        # Remove multiple punctuation
        text = re.sub(r"([.!?])\1+", r"\1", text)

        # Ensure space after punctuation
        text = re.sub(r"([.!?,;:])(?=[^\s\d])", r"\1 ", text)

        return text

    @classmethod
    def from_config(cls, config_dict: dict) -> Self:
        """Create normalizer from configuration dictionary."""
        config = MongolianTextNormalizerConfig(**config_dict)
        return cls(config=config)


def normalize_text(
    text: str,
    expand_abbreviations: bool = True,
    convert_numbers: bool = True,
) -> str:
    """Convenience function for text normalization.

    Args:
        text: Input text.
        expand_abbreviations: Whether to expand abbreviations.
        convert_numbers: Whether to convert numbers to words.

    Returns:
        Normalized text.
    """
    config = MongolianTextNormalizerConfig(
        expand_abbreviations=expand_abbreviations,
        convert_numbers=convert_numbers,
    )
    normalizer = MongolianTextNormalizer(config)
    return normalizer.normalize(text)
