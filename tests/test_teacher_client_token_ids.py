from scripts.opd import teacher_client


def test_exact_token_ids_are_sent_and_sliced(monkeypatch):
    captured = {}

    def fake_request(method, url, *, payload, timeout, retries):
        captured.update(payload)
        return {
            "choices": [
                {
                    "prompt_logprobs": [
                        None,
                        {11: {"logprob": -0.1}},
                        {20: {"logprob": -0.2}},
                        {21: {"logprob": -0.3}},
                    ]
                }
            ]
        }

    monkeypatch.setattr(teacher_client, "_request_json", fake_request)
    values = teacher_client.score_completion_token_logprobs(
        "http://teacher", "teacher", [10, 11], [20, 21]
    )
    assert captured["prompt"] == [10, 11, 20, 21]
    assert values == [-0.2, -0.3]


def test_compatibility_wrapper_does_not_retokenize_concatenated_boundary(monkeypatch):
    captured = {}

    class BoundaryTokenizer:
        def encode(self, text, add_special_tokens=False):
            return {"a": [1], "b": [2], "ab": [99]}[text]

    def fake_score(base_url, model, prompt_ids, completion_ids, **kwargs):
        captured["prompt"] = prompt_ids
        captured["completion"] = completion_ids
        return [-1.0]

    monkeypatch.setattr(teacher_client, "score_completion_token_logprobs", fake_score)
    teacher_client.score_completion_logprobs(
        "http://teacher", "teacher", "a", "b", BoundaryTokenizer()
    )
    assert captured == {"prompt": [1], "completion": [2]}


def test_teacher_token_id_mismatch_fails_closed(monkeypatch):
    def fake_request(method, url, *, payload, timeout, retries):
        return {
            "choices": [
                {
                    "prompt_logprobs": [
                        None,
                        {11: {"logprob": -0.1}},
                        {999: {"logprob": -0.2}},
                    ]
                }
            ]
        }

    monkeypatch.setattr(teacher_client, "_request_json", fake_request)
    try:
        teacher_client.score_completion_token_logprobs(
            "http://teacher", "teacher", [10, 11], [20]
        )
    except ValueError as exc:
        assert "token_id=20" in str(exc)
    else:
        raise AssertionError("mismatched teacher token ID should fail closed")


def test_plain_token_logprobs_fallback_is_rejected(monkeypatch):
    def fake_request(method, url, *, payload, timeout, retries):
        return {"choices": [{"logprobs": {"token_logprobs": [None, -0.1, -0.2]}}]}

    monkeypatch.setattr(teacher_client, "_request_json", fake_request)
    try:
        teacher_client.score_completion_token_logprobs(
            "http://teacher", "teacher", [10, 11], [20]
        )
    except RuntimeError as exc:
        assert "cannot prove exact" in str(exc)
    else:
        raise AssertionError("unidentified token logprobs should fail closed")


def test_unkeyed_prompt_logprob_scalar_is_rejected(monkeypatch):
    def fake_request(method, url, *, payload, timeout, retries):
        return {
            "choices": [
                {
                    "prompt_logprobs": [
                        None,
                        {11: {"logprob": -0.1}},
                        -0.2,
                    ]
                }
            ]
        }

    monkeypatch.setattr(teacher_client, "_request_json", fake_request)
    try:
        teacher_client.score_completion_token_logprobs(
            "http://teacher", "teacher", [10, 11], [20]
        )
    except ValueError as exc:
        assert "cannot prove identity" in str(exc)
        assert "token_id=20" in str(exc)
    else:
        raise AssertionError("unkeyed scalar prompt logprob should fail closed")
