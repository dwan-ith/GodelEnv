"""
Deterministic code repair task.
"""
from __future__ import annotations

import random
from typing import Optional

from godel_engine.code_eval import extract_code, parse_code_features, run_code_tests
from godel_engine.tasks.base import BaseTask, TaskInstance


_CODE_DATASET = [
    {
        "id": "code01",
        "prompt": "Write a Python function `is_palindrome(s)` that ignores spaces, punctuation, and case to check if a string is a palindrome.",
        "initial_solution": "def is_palindrome(s):\n    return s == s[::-1]",
        "function_name": "is_palindrome",
        "test_cases": [
            ("racecar", True),
            ("A man a plan a canal Panama", True),
            ("hello", False),
            ("No 'x' in Nixon", True),
        ],
    },
    {
        "id": "code02",
        "prompt": "Write a Python function `fibonacci(n)` that returns the nth Fibonacci number efficiently.",
        "initial_solution": "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n - 1) + fibonacci(n - 2)",
        "function_name": "fibonacci",
        "test_cases": [
            (0, 0),
            (1, 1),
            (5, 5),
            (10, 55),
            (20, 6765),
            (35, 9227465),
        ],
    },
]


class CodeImprovementTask(BaseTask):
    def __init__(self) -> None:
        super().__init__("code_improvement", "medium")
        self.dataset = _CODE_DATASET

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
            "syntax": "Is the Python code syntactically valid and safe to execute?",
            "tests": "Does the function pass the hidden test cases?",
            "documentation": "Does the function include a useful docstring?",
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
            scores = {"syntax": 0.0, "tests": 0.0, "documentation": 0.0}
            feedback = {
                "syntax": f"Syntax error: {features['syntax_error']}",
                "tests": "Tests were skipped because the code does not parse.",
                "documentation": "Add a valid function definition with a docstring.",
            }
            return await self._finalize_grade(
                task_instance,
                solution,
                0.0,
                scores,
                feedback,
            )

        syntax_score = 1.0
        test_result = run_code_tests(
            code,
            function_name=reference["function_name"],
            test_cases=reference["test_cases"],
            timeout_s=2.0,
        )
        test_score = test_result["passed"] / max(test_result["total"], 1)
        doc_score = 1.0 if features["has_docstring"] else 0.0

        scores = {
            "syntax": syntax_score,
            "tests": test_score,
            "documentation": doc_score,
        }
        total = 0.2 * syntax_score + 0.6 * test_score + 0.2 * doc_score

        feedback = {
            "syntax": "Syntax and safety checks passed.",
            "tests": (
                f"Passed {test_result['passed']}/{test_result['total']} tests."
                if not test_result["error"]
                else f"Passed {test_result['passed']}/{test_result['total']} tests. {test_result['error']}"
            ),
            "documentation": (
                "Docstring present."
                if doc_score >= 1.0
                else "Add a short docstring describing the function."
            ),
        }

        return await self._finalize_grade(task_instance, solution, total, scores, feedback)
