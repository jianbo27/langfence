# Privacy Notes

This project can sit directly in the request path for prompts and model outputs, so
privacy controls must be part of normal design rather than a later add-on.

## Defaults

- No telemetry.
- No provider credentials in config files or examples.
- CLI diagnostics redact prompt/message content and common secret keys by default.
- The proxy hides raw provider error bodies by default because providers may echo request
  fragments in error messages.
- Language detection is local-only in the built-in implementations.

## Data That Can Be Sensitive

- User prompts and system prompts.
- Model outputs, especially extracted JSON fields.
- Provider URLs when they identify private infrastructure.
- Authorization headers, API keys, cookies, tokens, and secrets.
- Validation failures if they include raw payload excerpts.

## Engineering Rules

- Do not add telemetry without an explicit opt-in flag.
- Do not log full request or response bodies in library code.
- Keep redaction on by default for human-facing diagnostics.
- Make any raw-output debug mode visibly named, for example `--show-sensitive`.
- Prefer returning issue codes and confidence scores over snippets of user content.
- Keep examples synthetic and free of real prompts, keys, endpoints, or customer data.
