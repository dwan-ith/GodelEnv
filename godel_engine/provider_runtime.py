from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


load_dotenv(override=False)

DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_ROUTER_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_HF_ROUTER_BASE_URL = "https://router.huggingface.co/v1"
DEFAULT_PROVIDER_ORDER = ("custom", "openai", "huggingface")


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


def _build_custom_provider() -> ProviderConfig | None:
    api_key = _env("API_KEY")
    base_url = _env("CUSTOM_API_BASE_URL") or _legacy_base_url()
    if not api_key or not base_url:
        return None
    model_name = _env("CUSTOM_MODEL_NAME") or _global_model_name() or DEFAULT_ROUTER_MODEL
    return ProviderConfig(
        name="custom",
        api_key=api_key,
        base_url=base_url,
        model_name=model_name,
    )


def _build_openai_provider() -> ProviderConfig | None:
    api_key = _env("OPENAI_API_KEY")
    if not api_key:
        return None

    base_url = _env("OPENAI_API_BASE_URL")
    legacy_base_url = _legacy_base_url()
    if not base_url and legacy_base_url and not _env("API_KEY") and not _env("HF_TOKEN"):
        base_url = legacy_base_url

    model_name = _env("OPENAI_MODEL_NAME") or _global_model_name() or DEFAULT_OPENAI_MODEL
    return ProviderConfig(
        name="openai",
        api_key=api_key,
        base_url=base_url,
        model_name=model_name,
    )


def _build_huggingface_provider() -> ProviderConfig | None:
    api_key = _env("HF_TOKEN")
    if not api_key:
        return None

    legacy_base_url = _legacy_base_url()
    base_url = _env("HF_API_BASE_URL")
    if not base_url and legacy_base_url and "huggingface.co" in legacy_base_url.lower():
        base_url = legacy_base_url
    if not base_url:
        base_url = DEFAULT_HF_ROUTER_BASE_URL

    model_name = _env("HF_MODEL_NAME") or _global_model_name() or DEFAULT_ROUTER_MODEL
    return ProviderConfig(
        name="huggingface",
        api_key=api_key,
        base_url=base_url,
        model_name=model_name,
    )


def _provider_order() -> list[str]:
    raw = _env("GODEL_PROVIDER_ORDER")
    if not raw:
        return list(DEFAULT_PROVIDER_ORDER)

    order: list[str] = []
    for item in raw.split(","):
        name = item.strip().lower()
        if not name or name in order:
            continue
        order.append(name)

    for name in DEFAULT_PROVIDER_ORDER:
        if name not in order:
            order.append(name)
    return order


def load_provider_configs() -> list[ProviderConfig]:
    candidates = {
        "custom": _build_custom_provider(),
        "openai": _build_openai_provider(),
        "huggingface": _build_huggingface_provider(),
    }

    configs: list[ProviderConfig] = []
    for name in _provider_order():
        config = candidates.get(name)
        if config is not None:
            configs.append(config)
    return configs


def load_provider_config() -> ProviderConfig:
    configs = load_provider_configs()
    if configs:
        return configs[0]
    return ProviderConfig(
        name="none",
        api_key=None,
        base_url=None,
        model_name=_global_model_name() or DEFAULT_OPENAI_MODEL,
    )


def describe_provider_configs() -> list[dict[str, str | bool | None]]:
    return [
        {
            "name": config.name,
            "configured": bool(config.api_key),
            "base_url": config.base_url,
            "model_name": config.model_name,
            "disabled": ProviderCircuitBreaker.is_disabled(config.name),
            "disabled_reason": ProviderCircuitBreaker.reason(config.name),
        }
        for config in load_provider_configs()
    ]


class ProviderCircuitBreaker:
    """Disable repeated retries for only the providers that actually failed."""

    _disabled: dict[str, str] = {}

    @classmethod
    def is_disabled(cls, provider_name: str | None = None) -> bool:
        if provider_name is None:
            return bool(cls._disabled)
        return provider_name in cls._disabled

    @classmethod
    def reason(cls, provider_name: str | None = None) -> str | None:
        if provider_name is None:
            if not cls._disabled:
                return None
            return "; ".join(f"{name}: {reason}" for name, reason in sorted(cls._disabled.items()))
        return cls._disabled.get(provider_name)

    @classmethod
    def disable(cls, provider_name: str, reason: str) -> None:
        cls._disabled[provider_name] = reason

    @classmethod
    def reset(cls, provider_name: str | None = None) -> None:
        if provider_name is None:
            cls._disabled = {}
            return
        cls._disabled.pop(provider_name, None)

    @classmethod
    def record_failure(cls, provider_name: str, exc: Exception) -> str:
        message = f"{type(exc).__name__}: {exc}"
        lowered = message.lower()
        hard_fail_markers = (
            "apiconnectionerror",
            "connection error",
            "failed to establish a new connection",
            "max retries exceeded",
            "timed out",
            "timeout",
            "insufficient_quota",
            "rate limit",
            "unauthorized",
            "authentication",
            "forbidden",
            "unsupported parameter",
            "invalid_request_error",
            "unsupported_value",
            "model_not_found",
            "does not exist",
            "winerror 10013",
        )
        if any(marker in lowered for marker in hard_fail_markers):
            cls.disable(provider_name, message)
        return message
