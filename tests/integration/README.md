# Integration Tests

GPU-backed vLLM and SGLang integration tests are intentionally not enabled in
default CI. Use the CLI against local servers first:

```bash
langfence compile --provider vllm --contract examples/contract.zh.yaml
langfence compile --provider sglang --contract examples/contract.zh.yaml
```

Then run the proxy against a running OpenAI-compatible endpoint.
