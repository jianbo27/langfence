import json

from langfence import ChoiceConstraint, OutputContract, compile_request


def test_sglang_choice_snapshot() -> None:
    compiled = compile_request(
        "sglang",
        [{"role": "user", "content": "classify"}],
        OutputContract(format=ChoiceConstraint(["approved", "rejected"])),
        base_payload={"model": "m"},
    )

    assert json.dumps(compiled.payload, sort_keys=True) == json.dumps(
        {
            "extra_body": {"regex": "^(?:approved|rejected)$"},
            "messages": [{"role": "user", "content": "classify"}],
            "model": "m",
        },
        sort_keys=True,
    )
