"""Runtime policy for safely routing task families around a LoRA adapter."""
from __future__ import annotations

import json
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ContextManager


@dataclass(frozen=True)
class AdapterRoutingPolicy:
    """Select the trained adapter or untouched base weights by task family."""

    base_fallback_tasks: frozenset[str] = frozenset()
    task_regression_tolerance: float = 0.02

    @classmethod
    def from_model_dir(cls, model_dir: str | Path) -> "AdapterRoutingPolicy":
        manifest_path = Path(model_dir) / "routing.json"
        if not manifest_path.exists():
            return cls()

        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if payload.get("type") != "task_conditional_lora":
            raise ValueError(f"Unsupported adapter routing type in {manifest_path}")
        tasks = payload.get("base_fallback_tasks", [])
        if not isinstance(tasks, list) or not all(isinstance(task, str) for task in tasks):
            raise ValueError(f"Invalid base_fallback_tasks in {manifest_path}")
        return cls(
            base_fallback_tasks=frozenset(tasks),
            task_regression_tolerance=float(
                payload.get("task_regression_tolerance", 0.02)
            ),
        )

    def route_for(self, task_type: str) -> str:
        return "base_fallback" if task_type in self.base_fallback_tasks else "trained_adapter"

    def model_context(self, model: Any, task_type: str) -> ContextManager[Any]:
        if task_type not in self.base_fallback_tasks:
            return nullcontext()
        if not hasattr(model, "disable_adapter"):
            raise TypeError("Base fallback routing requires a PEFT model with disable_adapter()")
        return model.disable_adapter()

