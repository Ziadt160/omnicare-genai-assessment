"""LLM providers.

Groq, NVIDIA NIM, GitHub Models and Ollama are all OpenAI-compatible. That is
worth stating plainly because it looks like four integrations and is one: the
same ChatOpenAI client with a different base_url and model. Adding a fifth
hosted provider is a row in this table, not a new adapter.

The fallback chain is what makes the free tier usable. Groq's limits are
generous but real, and an eval run firing thirty cases will find them; when the
primary trips its breaker, the secondary answers instead of the whole turn
failing.
"""

from __future__ import annotations

from dataclasses import dataclass

from libs.resilience.policy import CircuitBreaker, RetryPolicy

# base_url, default model, whether an API key is required.
PROVIDERS: dict[str, tuple[str, str, bool]] = {
    "groq": (
        "https://api.groq.com/openai/v1",
        # Verified against the live catalogue: llama-3.3-70b-versatile is gone.
        # Groq rotates its model list, so check /v1/models before assuming an
        # id still exists - a stale default fails as a 404 at the first tool
        # call, which reads like a code bug rather than a config one.
        "openai/gpt-oss-120b",
        True,
    ),
    "nvidia": (
        "https://integrate.api.nvidia.com/v1",
        "meta/llama-3.3-70b-instruct",
        True,
    ),
    "github": (
        "https://models.inference.ai.azure.com",
        "gpt-4o-mini",
        True,
    ),
    # Keyless. Not a model - a keyword router that answers from tool output, so
    # `docker compose up` demonstrates the whole system with no credential at
    # all. Never a default; see libs/adapters/llm_scripted.py.
    "fake": ("", "scripted", False),
    "ollama": (
        # Ollama exposes an OpenAI-compatible surface at /v1, so local runs the
        # same code path as hosted - no second client, no divergent behaviour.
        "http://ollama:11434/v1",
        # qwen2.5 is the most reliable small model for tool calling. Most other
        # 7B models silently return prose where a tool call was required, which
        # is a very expensive thing to discover on day three. Verified against
        # Ollama 0.21: emits a well-formed tool call with the right argument.
        "qwen2.5:7b",
        False,
    ),
}


@dataclass
class ProviderConfig:
    name: str
    model: str
    base_url: str
    api_key: str

    @property
    def requires_key(self) -> bool:
        return PROVIDERS[self.name][2]


def resolve(
    name: str, model: str = "", api_key: str = "", base_url: str = ""
) -> ProviderConfig:
    if name not in PROVIDERS:
        raise ValueError(
            f"Unknown provider {name!r}. Known: {', '.join(sorted(PROVIDERS))}"
        )
    default_url, default_model, needs_key = PROVIDERS[name]
    if needs_key and not api_key:
        raise ValueError(
            f"Provider {name!r} needs an API key. Set AGENT_LLM_API_KEY, or use "
            f"LLM_PROVIDER=ollama with `docker compose --profile local up`."
        )
    return ProviderConfig(
        name=name,
        model=model or default_model,
        base_url=base_url or default_url,
        api_key=api_key or "not-needed",
    )


def build_chat_model(config: ProviderConfig, temperature: float = 0.0, timeout: float = 20.0):
    """One client for every provider. Imported lazily so the tests that use
    FakeLLM never need langchain-openai installed."""
    if config.name == "fake":
        from libs.adapters.llm_scripted import ScriptedProvider

        return ScriptedProvider()

    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=config.model,
        base_url=config.base_url,
        api_key=config.api_key,
        temperature=temperature,
        timeout=timeout,
        max_retries=0,  # retry policy lives in libs.resilience, not the SDK
    )


def default_retry_policy(channel: str = "text") -> RetryPolicy:
    """Voice gets a much tighter deadline. A policyholder hearing four seconds
    of silence assumes the call dropped; the same wait in a chat window reads
    as thinking."""
    if channel == "voice":
        return RetryPolicy(attempts=2, base_delay_s=0.2, max_delay_s=1.0, deadline_s=6.0)
    return RetryPolicy(attempts=3, base_delay_s=0.25, max_delay_s=4.0, deadline_s=20.0)


def make_breaker(name: str) -> CircuitBreaker:
    return CircuitBreaker(name=name, failure_threshold=5, reset_after_s=30.0)
