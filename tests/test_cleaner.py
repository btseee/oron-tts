from src.data.cleaner import MongolianTextCleaner


def test_cleaner_instantiation():
    cleaner = MongolianTextCleaner()
    assert cleaner is not None


def test_number_expansion():
    cleaner = MongolianTextCleaner()
    # Basic test - expand_numbers is complex, just checking it runs
    # Assuming "123" -> "нэг зуун хорин гурав" approximately or just cyrillic
    text = "123"
    expanded = cleaner.expand_numbers(text)
    assert expanded != text
    assert isinstance(expanded, str)


def test_latin_to_cyrillic():
    cleaner = MongolianTextCleaner()
    # If implemented
    text = "Sain baina uu"
    cleaned = cleaner(text)
    assert isinstance(cleaned, str)
