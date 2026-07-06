from langfence.constraints import JsonSchemaConstraint
from langfence.serialization import load_contract


def test_load_contract_yaml() -> None:
    contract = load_contract("examples/contract.zh.yaml")

    assert isinstance(contract.format, JsonSchemaConstraint)
    assert contract.format.name == "localized_answer"
    assert contract.language is not None
    assert contract.language.include == ("zh",)
