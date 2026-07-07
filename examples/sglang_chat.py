from langfence import ChoiceConstraint, OutputContract, compile_request
from langfence.privacy import redact_for_display

contract = OutputContract(format=ChoiceConstraint(["approved", "rejected"]))

compiled = compile_request(
    provider="sglang",
    messages=[{"role": "user", "content": "Classify the request."}],
    contract=contract,
    base_payload={"model": "Qwen/Qwen3-8B", "temperature": 0},
)

print(redact_for_display(compiled.payload))
