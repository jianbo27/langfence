from langfence import (
    JsonSchemaConstraint,
    LanguagePolicy,
    OutputContract,
    compile_request,
)
from langfence.privacy import redact_for_display

contract = OutputContract(
    format=JsonSchemaConstraint(
        name="answer",
        schema={
            "type": "object",
            "properties": {
                "answer": {"type": "string"},
                "language": {"type": "string", "enum": ["zh"]},
            },
            "required": ["answer", "language"],
            "additionalProperties": False,
        },
    ),
    language=LanguagePolicy(include=["zh"], exclude=["en"], action="fail"),
)

compiled = compile_request(
    provider="vllm",
    messages=[{"role": "user", "content": "用中文解释 vLLM structured outputs。"}],
    contract=contract,
    base_payload={"model": "Qwen/Qwen3-8B", "temperature": 0},
)

print(redact_for_display(compiled.payload))
