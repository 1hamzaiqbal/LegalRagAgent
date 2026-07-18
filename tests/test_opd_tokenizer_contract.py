import sys

import pytest

from scripts.opd_math.tokenizer_contract import comparable_contract, main, tokenizer_fingerprint


class FakeBackend:
    def __init__(self, payload):
        self.payload = payload

    def to_str(self):
        return self.payload


class FakeTokenizer:
    bos_token_id = 1
    eos_token_id = 2
    pad_token_id = 2
    unk_token_id = 0
    sep_token_id = None
    cls_token_id = None
    mask_token_id = None
    chat_template = "template"

    def __init__(self, backend_payload):
        self.backend_tokenizer = FakeBackend(backend_payload)

    def get_vocab(self):
        return {"<unk>": 0, "a": 1, "b": 2}

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, enable_thinking):
        assert tokenize is False and add_generation_prompt is True
        return ("think:" if enable_thinking else "answer:") + messages[0]["content"]

    def encode(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        return [ord(char) % 17 for char in text]


def test_backend_normalizer_or_pretokenizer_drift_changes_contract():
    left = tokenizer_fingerprint(FakeTokenizer('{"normalizer":"A","pre_tokenizer":"P"}'))
    right = tokenizer_fingerprint(FakeTokenizer('{"normalizer":"B","pre_tokenizer":"P"}'))
    assert left["vocab_sha256"] == right["vocab_sha256"]
    assert left["backend_tokenizer_json_sha256"] != right["backend_tokenizer_json_sha256"]
    assert comparable_contract(left) != comparable_contract(right)


def test_slow_tokenizer_without_serializable_backend_fails_closed():
    tokenizer = FakeTokenizer("backend")
    tokenizer.backend_tokenizer = None
    with pytest.raises(ValueError, match="fast-tokenizer backend"):
        tokenizer_fingerprint(tokenizer)


def test_existing_contract_output_fails_before_model_loading(tmp_path, monkeypatch):
    output = tmp_path / "tokenizer_contract.json"
    output.write_text("existing\n")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "tokenizer_contract.py",
            "--teacher",
            "teacher",
            "--student",
            "student",
            "--output",
            str(output),
        ],
    )
    with pytest.raises(FileExistsError, match="refusing to overwrite tokenizer contract"):
        main()
    assert output.read_text() == "existing\n"
