"""
Deterministic Python optimization task.
"""
from __future__ import annotations

import random
from typing import Optional

from godel_engine.code_eval import extract_code, parse_code_features, run_code_tests
from godel_engine.tasks.base import BaseTask, TaskInstance


_PYTHON_DATASET = [
    {
        "id": "py01",
        "prompt": "Optimize this Python script that finds prime numbers and add comprehensive docstrings and type hints.",
        "initial_solution": (
            "import math\n"
            "def get_primes(n):\n"
            "    primes = []\n"
            "    for i in range(2, n):\n"
            "        is_prime = True\n"
            "        for j in range(2, i):\n"
            "            if i % j == 0:\n"
            "                is_prime = False\n"
            "                break\n"
            "        if is_prime:\n"
            "            primes.append(i)\n"
            "    return primes\n"
        ),
        "function_name": "get_primes",
        "test_cases": [
            (0, []),
            (2, []),
            (10, [2, 3, 5, 7]),
            (20, [2, 3, 5, 7, 11, 13, 17, 19]),
        ],
    }
]


class PythonOptimizedTask(BaseTask):
    def __init__(self) -> None:
        super().__init__("python_optimized", "medium")
        self.dataset = _PYTHON_DATASET

    def sample(
        self,
        rng: Optional[random.Random] = None,
        task_id: Optional[str] = None,
    ) -> TaskInstance:
        data = self._pick_dataset_entry(self.dataset, rng=rng, task_id=task_id)
        return TaskInstance(
            task_id=data["id"],
            difficulty=self.difficulty,
            prompt=data["prompt"],
            initial_solution=data["initial_solution"],
            reference=data,
        )

    def _get_rubrics(self) -> dict[str, str]:
        return {
            "correctness": "Does the optimized function still return the correct primes?",
            "efficiency": "Does the implementation show a meaningful optimization over the naive baseline?",
            "documentation": "Does the code include docstrings and type hints?",
        }

    async def grade(
        self,
        task_instance: TaskInstance,
        solution: str,
    ) -> tuple[float, dict[str, float], dict[str, str]]:
        reference = task_instance.reference
        code = extract_code(solution)
        features = parse_code_features(code)

        if not features["syntax_ok"]:
            scores = {"correctness": 0.0, "efficiency": 0.0, "documentation": 0.0}
            feedback = {
                "correctness": "The code does not parse, so correctness cannot be checked.",
                "efficiency": "Start with valid Python before optimizing.",
                "documentation": f"Syntax error: {features['syntax_error']}",
            }
            return await self._finalize_grade(
                task_instance,
                solution,
                0.0,
                scores,
                feedback,
            )

        test_result = run_code_tests(
            code,
            function_name=reference["function_name"],
            test_cases=reference["test_cases"],
            timeout_s=2.0,
        )
        correctness = test_result["passed"] / max(test_result["total"], 1)

        normalized = code.lower()
        has_fast_pattern = any(
            marker in normalized
            for marker in ["isqrt", "sqrt(", "p * p <=", "candidate * candidate <="]
        )
        mentions_sieve = "sieve" in normalized
        avoids_naive_loop = "for j in range(2, i)" not in normalized
        edge_case_guard = "n < 2" in normalized or "if n <= 2" in normalized

        efficiency = 0.25
        if avoids_naive_loop:
            efficiency += 0.25
        if has_fast_pattern:
            efficiency += 0.25
        if mentions_sieve:
            efficiency += 0.25
        if edge_case_guard:
            efficiency = min(1.0, efficiency + 0.1)
        efficiency = min(1.0, efficiency)

        documentation = 0.0
        if features["has_docstring"]:
            documentation += 0.5
        if features["has_type_hints"]:
            documentation += 0.5

        scores = {
            "correctness": correctness,
            "efficiency": efficiency,
            "documentation": documentation,
        }
        total = 0.5 * correctness + 0.3 * efficiency + 0.2 * documentation

        feedback = {
            "correctness": (
                f"Passed {test_result['passed']}/{test_result['total']} tests."
                if not test_result["error"]
                else f"Passed {test_result['passed']}/{test_result['total']} tests. {test_result['error']}"
            ),
            "efficiency": (
                "Use a tighter primality bound or a sieve-style approach."
                if efficiency < 0.95
                else "The implementation shows a real optimization over the baseline."
            ),
            "documentation": (
                "Add both docstrings and type hints."
                if documentation < 1.0
                else "Docstrings and type hints are present."
            ),
        }

        return await self._finalize_grade(task_instance, solution, total, scores, feedback)
