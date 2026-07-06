from langfence.privacy import REDACTED, redact_for_display


def test_redacts_message_content() -> None:
    payload = {
        "messages": [
            {"role": "user", "content": "private prompt"},
            {"role": "assistant", "content": "private answer"},
        ],
        "response_format": {"type": "json_schema"},
    }

    redacted = redact_for_display(payload)

    assert redacted["messages"][0]["content"] == REDACTED
    assert redacted["messages"][1]["content"] == REDACTED
    assert redacted["response_format"] == {"type": "json_schema"}


def test_redacts_secret_keys() -> None:
    payload = {
        "Authorization": "Bearer secret",
        "nested": {"api_key": "sk-secret"},
    }

    redacted = redact_for_display(payload)

    assert redacted["Authorization"] == REDACTED
    assert redacted["nested"]["api_key"] == REDACTED
