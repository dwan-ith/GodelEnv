"""
Gödel Engine — A Self-Improving RL Environment for Strategy Evolution

OpenEnv-compatible environment where an LLM agent iteratively improves
solutions through reinforcement learning, scored by LLM-as-a-judge graders.

Standalone usage (no web server required):
    from godel_engine import GodelEnvironment, AutoAgent
    env = GodelEnvironment()
    result = await env.reset(task_type="factual_qa")
    result = await env.step(action)
"""

__version__ = "0.2.0"

from godel_engine.models import (
    GodelAction,
    GodelObservation,
    GodelState,
    GodelStepResult,
    EditType,
)
from godel_engine.environment import GodelEnvironment
from godel_engine.agent import AutoAgent
from godel_engine.client import GodelEngineEnv

__all__ = [
    "GodelEnvironment",
    "AutoAgent",
    "GodelAction",
    "GodelObservation",
    "GodelState",
    "GodelStepResult",
    "EditType",
    "GodelEngineEnv",
]
