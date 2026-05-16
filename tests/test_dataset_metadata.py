from src.data.dataset import _attr_tokens_from_metadata


def test_attr_tokens_from_common_voice_metadata() -> None:
    item = {"gender": "female", "age": "twenties"}

    tokens = _attr_tokens_from_metadata(item, gender_column="gender", age_column="age")

    assert tokens == ["[FEMALE]", "[YOUNG]"]


def test_attr_tokens_ignore_unknown_metadata() -> None:
    item = {"gender": "other", "age": "unknown"}

    tokens = _attr_tokens_from_metadata(item, gender_column="gender", age_column="age")

    assert tokens == []
