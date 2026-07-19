"""
Provider runtime configuration for GodelEnv hybrid mode.

GodelEnv uses a hybrid execution model:
1. **LLM-first (default)**: Tries configured LLM providers in order
2. **Deterministic fallback**: If all LLM providers fail, uses heuristic policy

Provider priority (configurable via GODEL_PROVIDER_ORDER):
1. huggingface - HF Router / Inference API
2. ollama - Local Ollama instance
3. custom - Any OpenAI-compatible endpoint (vLLM, etc.)
4. openai - OpenAI API

To use a local model (Ollama):
    OLLAMA_API_BASE_URL=http://localhost:11434/v1
    OLLAMA_MODEL_NAME=qwen2.5:7b

To use a local model (vLLM):
    API_BASE_URL=http://localhost:8000/v1
    API_KEY=dummy  # vLLM often doesn't need real auth
    MODEL_NAME=Qwen/Qwen2.5-7B-Instruct
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import httpx
from openai import AsyncOpenAI
from dotenv import load_dotenv


load_dotenv(override=False)
logger = logging.getLogger("godel_env.provider_runtime")

# Flags to avoid spamming repeated log messages
_warned_no_providers = False
_logged_provider_info = False

DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_ROUTER_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_OLLAMA_MODEL = "qwen2.5:7b"
DEFAULT_HF_ROUTER_BASE_URL = "https://router.huggingface.co/v1"
DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434/v1"
DEFAULT_PROVIDER_ORDER = ("huggingface", "ollama", "custom", "openai")


@dataclass(frozen=True)
class ProviderConfig:
    name: str
    api_key: str | None
    base_url: str | None
    model_name: str


def _env(name: str) -> str | None:
    value = os.getenv(name)
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def _legacy_base_url() -> str | None:
    return _env("API_BASE_URL")


def _global_model_name() -> str | None:
    return _env("MODEL_NAME")


def _looks_like_huggingface_hub_model(name: str) -> bool:
    """
    OpenAI model IDs are like gpt-4o-mini, o1, o1-mini.
    Hub-style IDs are org/model, e.g. Qwen/Qwen2.5-7B-Instruct.
    If MODEL_NAME is a Hub ID, it must not be sent to the OpenAI API.
    """
    if not name or "/" not in name:
        return False
    if name.lower().startswith("ft:") or "ft:" in name[:4]:
        return False
    return True


def _openai_model_name() -> str:
    """
    Resolve the model name for the OpenAI *official* API only.

    Never use HuggingFace Hub IDs (Qwen/..., meta-llama/...) from MODEL_NAME
    for OpenAI — that returns 400 invalid model ID. Use OPENAI_MODEL_NAME or
    the default, or a plain MODEL_NAME like gpt-4o-mini (no org/model slash).
    """
    explicit = _env("OPENAI_MODEL_NAME")
    if explicit:
        return explicit
    g = _global_model_name()
    if g and not _looks_like_huggingface_hub_model(g):
        return g
    if g and _looks_like_huggingface_hub_model(g):
        logger.info(
            "OpenAI: MODEL_NAME=%r looks like a HuggingFace Hub id; using %s for OpenAI API. "
            "Set OPENAI_MODEL_NAME (e.g. gpt-4o-mini) if you use OpenAI as fallback.",
            g,
            DEFAULT_OPENAI_MODEL,
        )
    return DEFAULT_OPENAI_MODEL


def _first_env(*names: str) -> str | None:
    for name in names:
        value = _env(name)
        if value:
            return value
    return None


def _role_model_env(order_env: str) -> str | None:
    if order_env == "GODEL_AGENT_PROVIDER_ORDER":
        return _env("GODEL_AGENT_MODEL_NAME")
    if order_env == "GODEL_VERIFIER_PROVIDER_ORDER":
        return _env("GODEL_VERIFIER_MODEL_NAME")
    return None


def _build_custom_provider(order_env: str = "GODEL_PROVIDER_ORDER") -> ProviderConfig | None:
    api_key = _env("API_KEY") or _env("CUSTOM_API_KEY") or _env("OPENROUTER_API_KEY")
    base_url = (
        _env("CUSTOM_API_BASE_URL")
        or _env("OPENROUTER_API_BASE_URL")
        or _legacy_base_url()
        or (DEFAULT_OPENROUTER_BASE_URL if _env("OPENROUTER_API_KEY") else None)
    )
    if not api_key or not base_url:
        return None
    model_name = (
        _role_model_env(order_env)
        or _env("CUSTOM_MODEL_NAME")
        or _env("OPENROUTER_MODEL_NAME")
        or _global_model_name()
        or DEFAULT_ROUTER_MODEL
    )
    return ProviderConfig(
        name="custom",
        api_key=api_key,
        base_url=base_url,
        model_name=model_name,
    )


def _build_openai_provider(order_env: str = "GODEL_PROVIDER_ORDER") -> ProviderConfig | None:
    api_key = _env("OPENAI_API_KEY")
    if not api_key:
        return None

    base_url = _env("OPENAI_API_BASE_URL")
    legacy_base_url = _legacy_base_url()
    if not base_url and legacy_base_url and not _env("API_KEY") and not _env("HF_TOKEN"):
        base_url = legacy_base_url

    model_name = _role_model_env(order_env) or _openai_model_name()
    return ProviderConfig(
        name="openai",
        api_key=api_key,
        base_url=base_url,
        model_name=model_name,
    )


def _build_huggingface_provider() -> ProviderConfig | None:
    api_key = _first_env(
        "HF_TOKEN",
        "HF_API_KEY",
        "HUGGINGFACE_API_KEY",
        "HUGGINGFACE_TOKEN",
        "HUGGINGFACEHUB_API_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
        "HF_ACCESS_TOKEN",
    )
    if not api_key:
        return None

    # In HF Spaces it is common to configure just HF_TOKEN + API_BASE_URL +
    # MODEL_NAME as secrets. If HF_TOKEN is present, treat API_BASE_URL as the
    # HF-compatible endpoint unless HF_API_BASE_URL is explicitly set.
    base_url = _env("HF_API_BASE_URL") or _legacy_base_url() or DEFAULT_HF_ROUTER_BASE_URL

    model_name = (
        _env("HF_MODEL_NAME")
        or _env("HF_INFERENCE_MODEL")
        or _global_model_name()
        or DEFAULT_ROUTER_MODEL
    )
    return ProviderConfig(
        name="huggingface",
        api_key=api_key,
        base_url=base_url,
        model_name=model_name,
    )


def _build_ollama_provider() -> ProviderConfig | None:
    """
    Build Ollama provider for local model inference.
    
    Ollama runs locally and provides an OpenAI-compatible API.
    No API key is required for local instances.
    
    Configuration:
        OLLAMA_API_BASE_URL=http://localhost:11434/v1
        OLLAMA_MODEL_NAME=qwen2.5:7b
    """
    base_url = _env("OLLAMA_API_BASE_URL") or _env("OLLAMA_HOST")
    
    # Also check if the default Ollama endpoint is accessible
    if not base_url:
        # Check if GODEL_USE_OLLAMA is set to enable auto-detection
        if _env("GODEL_USE_OLLAMA") or _env("OLLAMA_MODEL_NAME"):
            base_url = DEFAULT_OLLAMA_BASE_URL
    
    if not base_url:
        return None
    
    # Ollama doesn't require an API key for local access
    api_key = _env("OLLAMA_API_KEY") or "ollama"  # Placeholder for OpenAI client
    
    model_name = (
        _env("OLLAMA_MODEL_NAME")
        or _env("OLLAMA_MODEL")
        or DEFAULT_OLLAMA_MODEL
    )
    
    return ProviderConfig(
        name="ollama",
        api_key=api_key,
        base_url=base_url,
        model_name=model_name,
    )


def _provider_order(env_name: str = "GODEL_PROVIDER_ORDER") -> list[str]:
    raw = _env(env_name) or _env("GODEL_PROVIDER_ORDER")
    if not raw:
        return list(DEFAULT_PROVIDER_ORDER)

    order: list[str] = []
    for item in raw.split(","):
        name = item.strip().lower()
        if not name or name in order:
            continue
        order.append(name)

    return order


def load_provider_configs(*, order_env: str = "GODEL_PROVIDER_ORDER") -> list[ProviderConfig]:
    """
    Load all configured LLM providers in priority order.
    
    Returns providers in order of preference:
    1. huggingface - HF Router / Inference API
    2. ollama - Local Ollama instance  
    3. custom - Any OpenAI-compatible endpoint (vLLM, etc.)
    4. openai - OpenAI API
    
    Customize order via GODEL_PROVIDER_ORDER env var.
    """
    global _warned_no_providers, _logged_provider_info
    
    candidates = {
        "custom": _build_custom_provider(order_env),
        "openai": _build_openai_provider(order_env),
        "huggingface": _build_huggingface_provider(),
        "ollama": _build_ollama_provider(),
    }

    configs: list[ProviderConfig] = []
    for name in _provider_order(order_env):
        config = candidates.get(name)
        if config is not None:
            configs.append(config)
    
    # Log provider info only once per session to avoid spam
    if configs:
        if not _logged_provider_info:
            logger.info(
                "Configured LLM providers (in priority order): %s",
                ", ".join(f"{c.name}:{c.model_name}" for c in configs)
            )
            _logged_provider_info = True
    else:
        if not _warned_no_providers:
            logger.warning(
                "No LLM providers configured. Set HF_TOKEN, OPENAI_API_KEY, or "
                "OLLAMA_MODEL_NAME to enable LLM mode."
            )
            _warned_no_providers = True
    
    return configs


def load_provider_config(*, order_env: str = "GODEL_PROVIDER_ORDER") -> ProviderConfig:
    configs = load_provider_configs(order_env=order_env)
    if configs:
        return configs[0]
    return ProviderConfig(
        name="none",
        api_key=None,
        base_url=None,
        model_name=_global_model_name() or DEFAULT_OPENAI_MODEL,
    )


def provider_completion_kwargs(provider_name: str) -> dict[str, Any]:
    """Return explicit, provider-specific chat-completion options.

    Ollama normally chooses its accelerator automatically. ``OLLAMA_NUM_GPU``
    lets deployments force CPU inference (0) or a specific GPU-layer count
    without leaking Ollama-only fields to other OpenAI-compatible providers.
    """
    if provider_name != "ollama":
        return {}

    raw_num_gpu = _env("OLLAMA_NUM_GPU")
    if raw_num_gpu is None:
        return {}
    try:
        num_gpu = int(raw_num_gpu)
    except ValueError as exc:
        raise ValueError("OLLAMA_NUM_GPU must be a non-negative integer") from exc
    if num_gpu < 0:
        raise ValueError("OLLAMA_NUM_GPU must be a non-negative integer")
    return {"extra_body": {"options": {"num_gpu": num_gpu}}}


def _env_flag(name: str, default: bool) -> bool:
    value = _env(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


class _OllamaNativeCompletions:
    """Minimal Ollama adapter matching the AsyncOpenAI completion surface."""

    def __init__(self, base_url: str) -> None:
        root = base_url.rstrip("/")
        self.base_url = root[:-3] if root.endswith("/v1") else root

    async def create(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float | None = None,
        response_format: dict[str, str] | None = None,
        extra_body: dict[str, Any] | None = None,
        **_: Any,
    ) -> SimpleNamespace:
        options: dict[str, Any] = {"num_predict": max_tokens}
        if temperature is not None:
            options["temperature"] = temperature
        if extra_body and isinstance(extra_body.get("options"), dict):
            options.update(extra_body["options"])

        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
            # Reasoning-only tokens are not usable as environment actions.
            "think": _env_flag("OLLAMA_THINK", False),
            "options": options,
        }
        if response_format and response_format.get("type") == "json_object":
            body["format"] = "json"

        async with httpx.AsyncClient(timeout=None) as client:
            response = await client.post(f"{self.base_url}/api/chat", json=body)
            response.raise_for_status()
            payload = response.json()

        message = payload.get("message") or {}
        usage = SimpleNamespace(
            prompt_tokens=int(payload.get("prompt_eval_count", 0) or 0),
            completion_tokens=int(payload.get("eval_count", 0) or 0),
            total_tokens=int(payload.get("prompt_eval_count", 0) or 0)
            + int(payload.get("eval_count", 0) or 0),
        )
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=str(message.get("content", "")))
                )
            ],
            usage=usage,
        )


class _OllamaNativeClient:
    def __init__(self, base_url: str) -> None:
        self.chat = SimpleNamespace(completions=_OllamaNativeCompletions(base_url))


def build_provider_client(provider: ProviderConfig) -> Any:
    """Build a provider client while preserving Ollama's native controls."""
    if provider.name == "ollama":
        return _OllamaNativeClient(provider.base_url or DEFAULT_OLLAMA_BASE_URL)
    return AsyncOpenAI(
        api_key=provider.api_key,
        base_url=provider.base_url if provider.base_url else None,
    )


def describe_provider_configs(*, order_env: str = "GODEL_PROVIDER_ORDER") -> list[dict[str, str | bool | None]]:
    return [
        {
            "name": config.name,
            "configured": bool(config.api_key),
            "base_url": config.base_url,
            "model_name": config.model_name,
            "disabled": ProviderCircuitBreaker.is_disabled(config.name),
            "disabled_reason": ProviderCircuitBreaker.reason(config.name),
        }
        for config in load_provider_configs(order_env=order_env)
    ]


def describe_provider_environment() -> dict[str, bool]:
    """Return non-secret presence checks for supported provider env vars."""
    return {
        # Hugging Face
        "HF_TOKEN": bool(_env("HF_TOKEN")),
        "HF_API_KEY": bool(_env("HF_API_KEY")),
        "HUGGINGFACE_API_KEY": bool(_env("HUGGINGFACE_API_KEY")),
        "HUGGINGFACE_TOKEN": bool(_env("HUGGINGFACE_TOKEN")),
        "HUGGINGFACEHUB_API_TOKEN": bool(_env("HUGGINGFACEHUB_API_TOKEN")),
        "HUGGING_FACE_HUB_TOKEN": bool(_env("HUGGING_FACE_HUB_TOKEN")),
        "HF_ACCESS_TOKEN": bool(_env("HF_ACCESS_TOKEN")),
        # Custom / vLLM
        "API_KEY": bool(_env("API_KEY")),
        "API_BASE_URL": bool(_env("API_BASE_URL")),
        "OPENROUTER_API_KEY": bool(_env("OPENROUTER_API_KEY")),
        # OpenAI
        "OPENAI_API_KEY": bool(_env("OPENAI_API_KEY")),
        # Ollama (local)
        "OLLAMA_API_BASE_URL": bool(_env("OLLAMA_API_BASE_URL")),
        "OLLAMA_MODEL_NAME": bool(_env("OLLAMA_MODEL_NAME")),
        "OLLAMA_NUM_GPU": bool(_env("OLLAMA_NUM_GPU")),
        "OLLAMA_THINK": bool(_env("OLLAMA_THINK")),
        "GODEL_USE_OLLAMA": bool(_env("GODEL_USE_OLLAMA")),
    }


def get_active_provider() -> str | None:
    """Return the name of the first available LLM provider, or None."""
    configs = load_provider_configs()
    for config in configs:
        if config.api_key and not ProviderCircuitBreaker.is_disabled(config.name):
            return config.name
    return None


def is_llm_available() -> bool:
    """Check if any LLM provider is configured and available."""
    return get_active_provider() is not None


class ProviderCircuitBreaker:
    """Disable repeated retries without coupling independent runtime roles."""

    _disabled: dict[str, str] = {}

    @classmethod
    def _key(cls, provider_name: str, scope: str | None = None) -> str:
        return f"{scope}:{provider_name}" if scope else provider_name

    @classmethod
    def is_disabled(
        cls, provider_name: str | None = None, *, scope: str | None = None
    ) -> bool:
        if provider_name is None:
            return bool(cls._disabled)
        return cls._key(provider_name, scope) in cls._disabled

    @classmethod
    def reason(
        cls, provider_name: str | None = None, *, scope: str | None = None
    ) -> str | None:
        if provider_name is None:
            if not cls._disabled:
                return None
            return "; ".join(f"{name}: {reason}" for name, reason in sorted(cls._disabled.items()))
        return cls._disabled.get(cls._key(provider_name, scope))

    @classmethod
    def disable(
        cls, provider_name: str, reason: str, *, scope: str | None = None
    ) -> None:
        cls._disabled[cls._key(provider_name, scope)] = reason

    @classmethod
    def reset(
        cls, provider_name: str | None = None, *, scope: str | None = None
    ) -> None:
        global _warned_no_providers, _logged_provider_info
        if provider_name is None:
            cls._disabled = {}
            # Also reset the logging flags so warnings can appear again after reset
            _warned_no_providers = False
            _logged_provider_info = False
            return
        if scope is not None:
            cls._disabled.pop(cls._key(provider_name, scope), None)
            return
        cls._disabled.pop(provider_name, None)
        suffix = f":{provider_name}"
        cls._disabled = {
            key: value for key, value in cls._disabled.items() if not key.endswith(suffix)
        }

    @classmethod
    def record_failure(
        cls, provider_name: str, exc: Exception, *, scope: str | None = None
    ) -> str:
        message = f"{type(exc).__name__}: {exc}"
        lowered = message.lower()
        if provider_name == "huggingface" and (
            "unsupported parameter" in lowered
            or "response_format" in lowered
            or "json_object" in lowered
        ):
            # HF Router models vary in structured-output support. Callers avoid
            # JSON mode for HF, but if a provider still reports this, fall back
            # without permanently disabling the HF route.
            return message
        # Only disable the provider for likely-persistent failures (network, auth, quota).
        # Do NOT disable for: wrong model name, bad JSON from model, invalid request body — those
        # are often fixable config issues and would block the whole session after one bad call.
        hard_fail_markers = (
            "apiconnectionerror",
            "connection error",
            "failed to establish a new connection",
            "max retries exceeded",
            "timed out",
            "timeout",
            "insufficient_quota",
            "rate limit",
            "401",
            "403",
            "unauthorized",
            "forbidden",
            "winerror 10013",
        )
        soft_fail_markers = (
            "jsondecodeerror",
            "invalid model",
            "model_not_found",
            "invalid_request_error",
            "does not exist",
            "unsupported_value",
        )
        if any(s in lowered for s in soft_fail_markers):
            return message
        if any(marker in lowered for marker in hard_fail_markers):
            cls.disable(provider_name, message, scope=scope)
        return message
