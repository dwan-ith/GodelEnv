"""
Base task interface for Godel Env tasks.
"""
from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from godel_engine.graders.agent_grader import AgentGrader


class TaskInstance:
    """A concrete problem instance sampled from a task family."""

    def __init__(
        self,
        task_id: str,
        difficulty: str,
        prompt: str,
        initial_solution: str,
        reference: Any = None,
    ) -> None:
        self.task_id = task_id
        self.difficulty = difficulty
        self.prompt = prompt
        self.initial_solution = initial_solution
        self.reference = reference


class BaseTask(ABC):
    """Shared interface for all task families."""

    def __init__(self, name: str, difficulty: str = "easy") -> None:
        self.name = name
        self.difficulty = difficulty
        self.llm_grader = AgentGrader()
        self.last_grading_source = "deterministic"
        self.last_grading_error: str | None = None

    @abstractmethod
    def sample(
        self,
        rng: Optional[random.Random] = None,
        task_id: Optional[str] = None,
    ) -> TaskInstance:
        """Sample or fetch a problem instance."""

    @abstractmethod
    def _get_rubrics(self) -> Dict[str, str]:
        """Return rubric descriptions for the task."""

    @abstractmethod
    async def grade(
        self,
        task_instance: TaskInstance,
        solution: str,
    ) -> tuple[float, Dict[str, float], Dict[str, str]]:
        """Grade a candidate solution."""

    def _pick_dataset_entry(
        self,
        dataset: list[dict[str, Any]],
        rng: Optional[random.Random] = None,
        task_id: Optional[str] = None,
    ) -> dict[str, Any]:
        if task_id is not None:
            for item in dataset:
                if item["id"] == task_id:
                    return item
            raise KeyError(f"Unknown task_id `{task_id}` for task `{self.name}`")

        chooser = rng or random.Random()
        return chooser.choice(dataset)

    async def _finalize_grade(
        self,
        task_instance: TaskInstance,
        solution: str,
        deterministic_total: float,
        deterministic_scores: Dict[str, float],
        deterministic_feedback: Dict[str, str],
    ) -> tuple[float, Dict[str, float], Dict[str, str]]:
        llm_result = await self.llm_grader.safe_grade(
            task_prompt=task_instance.prompt,
            current_solution=solution,
            rubrics=self._get_rubrics(),
        )
        if llm_result is not None:
            llm_total, llm_scores, llm_feedback = llm_result
            self.last_grading_source = self.llm_grader.last_source or "llm"
            self.last_grading_error = None
            merged_feedback = {
                name: llm_feedback.get(name) or deterministic_feedback.get(name, "")
                for name in deterministic_scores
            }
            return llm_total, llm_scores, merged_feedback

        self.last_grading_source = "deterministic"
        self.last_grading_error = self.llm_grader.last_error
        return deterministic_total, deterministic_scores, deterministic_feedback
