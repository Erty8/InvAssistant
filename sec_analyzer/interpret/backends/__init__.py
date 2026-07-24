"""Pluggable LLM transports for the two-phase interpret flow.

Each backend exposes the same minimal contract the HTTP-API/Ollama transports
already follow inside :mod:`sec_analyzer.interpret.analyzer`: given a
``(system, user)`` prompt pair it returns the model's RAW reply text (the
schema JSON, possibly fenced), which the caller then feeds through the shared
``analyzer._parse_model_json`` fence-strip/parse step. Backends never parse the
analyzer's own JSON schema themselves; they only handle their own transport and
(for ``claude_code``) unwrap their transport envelope.
"""
