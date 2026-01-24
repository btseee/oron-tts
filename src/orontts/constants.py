"""Shared constants for OronTTS."""

# Audio constants
SAMPLE_RATE = 22050
HOP_LENGTH = 256
WIN_LENGTH = 1024
N_FFT = 1024
N_MELS = 80
MEL_FMIN = 0.0
MEL_FMAX = 8000.0

# Text constants
PAD = "_"
PUNCTUATION = "!'(),-.:;? "
MONGOLIAN_CYRILLIC = "абвгдеёжзийклмноөпрстуүфхцчшщъыьэюя"
SYMBOLS = [PAD] + list(PUNCTUATION) + list(MONGOLIAN_CYRILLIC.upper()) + list(MONGOLIAN_CYRILLIC)

# Symbol to ID mapping
SYMBOL_TO_ID = {s: i for i, s in enumerate(SYMBOLS)}
ID_TO_SYMBOL = {i: s for i, s in enumerate(SYMBOLS)}

# Phoneme constants (IPA subset for Mongolian via espeak-ng)
PHONEME_PAD = "_"
PHONEME_PUNCTUATION = "!'(),-.:;? "
# espeak-ng Mongolian phoneme set (approximation)
MONGOLIAN_PHONEMES = (
    "aɐeɛiɪoɔuʊəœøyʏ"  # Vowels
    "bpdtgkmnŋfvszʃʒxhrlj"  # Consonants
    "ːˈˌ"  # Length/stress markers
)
PHONEME_SYMBOLS = [PHONEME_PAD] + list(PHONEME_PUNCTUATION) + list(MONGOLIAN_PHONEMES)

PHONEME_TO_ID = {s: i for i, s in enumerate(PHONEME_SYMBOLS)}
ID_TO_PHONEME = {i: s for i, s in enumerate(PHONEME_SYMBOLS)}
