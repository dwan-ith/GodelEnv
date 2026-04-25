"""
Gödel Engine 2.0 — Recursive Self-Improvement Under Verifier-Backed Meta-Evaluation

An OpenEnv-compatible environment where the agent proposes, tests, and
selectively adopts self-modifications to its own reasoning strategies.

The core primitive is the StrategyPatch: the agent mutates its reasoning
policy, the environment evaluates parent vs child on held-out tasks, and
the Governor accepts or rejects based on multi-objective utility.

Standalone usage (no web server required):
    from godel_engine import GodelEnvironment, StrategyPatch, GodelAction
    env = GodelEnvironment()
    result = await env.reset(task_type="factual_qa")
    patch = StrategyPatch(improved_strategy="...", hypothesis="...")
    action = GodelAction(solution="...", strategy_patch=patch)
    result = await env.step(action)
    print(result.patch_decision)  # Governor's verdict
"""

__version__ = "2.0.0"

from godel_engine.models import (
    GodelAction,
    GodelObservation,
    GodelState,
    GodelStepResult,
    RewardBreakdown,
    EditType,
    StrategyPatch,
    PatchDecision,
)
from godel_engine.environment import GodelEnvironment
from godel_engine.agent import AutoAgent
from godel_engine.client import GodelEngineEnv
from godel_engine.guards import run_all_guards, run_strategy_guards, GuardResult
from godel_engine.heuristic_policy import build_heuristic_action, build_heuristic_solution
from godel_engine.curriculum import CurriculumController
from godel_engine.evolution import (
    Strategy,
    StrategyRegistry,
    Governor,
    GovernorConfig,
    HuxleyTracker,
)

__all__ = [
    # Core environment
    "GodelEnvironment",
    "AutoAgent",
    "GodelEngineEnv",
    # Actions & Observations
    "GodelAction",
    "GodelObservation",
    "GodelState",
    "GodelStepResult",
    "RewardBreakdown",
    "EditType",
    # GodelEnv 2.0: Strategy self-improvement
    "StrategyPatch",
    "PatchDecision",
    "Strategy",
    "StrategyRegistry",
    "Governor",
    "GovernorConfig",
    # Anti-hacking
    "run_all_guards",
    "run_strategy_guards",
    "GuardResult",
    # Supporting
    "CurriculumController",
    "HuxleyTracker",
    "build_heuristic_action",
    "build_heuristic_solution",
]
