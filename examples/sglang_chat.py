from langfence import ChoiceConstraint, OutputContract, compile_request

contract = OutputContract(format=ChoiceConstraint(["approved", "rejected"]))

compiled = compile_request(
    provider="sglang",
    messages=[{"role": "user", "content": "Classify the request."}],
    contract=contract,
    base_payload={"model": "Qwen/Qwen3-8B", "temperature": 0},
)

print(compiled.payload)
