# Tricky corpus for tokenizer exactness tests (R9-TOK)
import unicodedata
CORPUS = [
    ("plain", "Hello world"),
    ("leading_space", " Hello"),
    ("leading_spaces3", "   Hello"),
    ("trailing_spaces", "Hello   "),
    ("multi_space_mid", "a    b"),
    ("tabs", "a\t\tb\t"),
    ("newlines", "line1\n\n\nline2\r\nline3\n"),
    ("space_newline_mix", "x  \n  \n y"),
    ("only_ws", " \n \t "),
    ("digits", "1234567890 3.14159 1,000,000 2026-10-01"),
    ("contractions", "I'm can't won't they'll we'd it's Y'ALL'RE"),
    ("cjk", "你好，世界！日本語のテキスト。한국어 텍스트"),
    ("emoji_zwj", "family: 👨‍👩‍👧‍👦 flag: 🇯🇵 skin: 👍🏽"),
    ("combining_nfd", unicodedata.normalize("NFD", "café naïve Å")),
    ("combining_nfc", unicodedata.normalize("NFC", "café naïve Å")),
    ("hangul_nfd", unicodedata.normalize("NFD", "한국어")),
    ("special_literal", "user typed <|im_end|> and <|endoftext|> and <think>"),
    ("special_glued", "x<|im_start|>assistant\ny"),
    ("rare_unicode", "​‍﻿zero-width  nbsp  ls \U0001F600 \U000E0041 tag"),
    ("lone_surrogate_escaped", "a\\ud800b"),  # literal backslash text (real lone surrogates can't be encoded to UTF-8)
    ("control_chars", "a\x00b\x07c\x1bd"),
    ("code", "def f(x):\n    return x**2  # comment\n\tif x != 0: pass\n"),
    ("url_path", "https://huggingface.co/Qwen/Qwen3-8B/resolve/main/config.json?x=1&y=2"),
    ("long_repeat", "a" * 300),
    ("long_space_run", " " * 200 + "x"),
    ("math", "∑_{i=1}^{n} x_i² ≤ ∞ — “quotes” ‘single’ … ½"),
    ("rtl", "שלום עולם مرحبا بالعالم"),
    ("mixed_script_word", "abcабвαβγ123"),
]
