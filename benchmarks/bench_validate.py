from __future__ import annotations

import time

from langfence import JsonSchemaConstraint, OutputContract, validate_output


def main() -> None:
    contract = OutputContract(
        format=JsonSchemaConstraint(
            schema={
                "type": "object",
                "properties": {"answer": {"type": "string"}},
                "required": ["answer"],
            }
        )
    )
    sample = '{"answer": "ok"}'
    start = time.perf_counter()
    iterations = 10_000
    for _ in range(iterations):
        validate_output(sample, contract)
    elapsed = time.perf_counter() - start
    print({"iterations": iterations, "seconds": elapsed, "per_second": iterations / elapsed})


if __name__ == "__main__":
    main()
