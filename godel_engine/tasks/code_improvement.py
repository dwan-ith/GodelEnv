"""
Code Improvement Task (Medium difficulty).
"""
import random
import ast
import re
import asyncio
from typing import Optional

from godel_engine.tasks.base import BaseTask, TaskInstance

_CODE_DATASET = [
    {
        "id": "code01",
        "prompt": "Write a python function `is_palindrome(s)` that ignores spaces, punctuation, and case to check if a string is a palindrome.",
        "initial_solution": "def is_palindrome(s):\n    return s == s[::-1]",
        "test_cases": [
            ("racecar", True),
            ("A man a plan a canal Panama", True),
            ("hello", False),
            ("No 'x' in Nixon", True),
        ]
    },
    {
        "id": "code02",
        "prompt": "Write a python function `fibonacci(n)` that returns the nth Fibonacci number efficiently.",
        "initial_solution": "def fibonacci(n):\n    if n <= 1: return n\n    return fibonacci(n-1) + fibonacci(n-2)",
        "test_cases": [
            (0, 0), (1, 1), (5, 5), (10, 55), (20, 6765)
        ]
    }
]

class CodeImprovementTask(BaseTask):
    def __init__(self):
        super().__init__("code_improvement", "medium")
        self.dataset = _CODE_DATASET

    def sample(self, rng: Optional[random.Random] = None) -> TaskInstance:
        _rng = rng or random.Random()
        data = _rng.choice(self.dataset)
        return TaskInstance(
            task_id=data["id"],
            difficulty=self.difficulty,
            prompt=data["prompt"],
            initial_solution=data["initial_solution"],
            reference=data
        )
        
    def _extract_code(self, solution: str) -> str:
        pattern = r'```(?:python)?\s*\n?(.*?)```'
        match = re.search(pattern, solution, re.DOTALL)
        if match:
            return match.group(1).strip()
        return solution.strip()

    def _get_rubrics(self) -> dict[str, str]:
        return {
            "syntax": "Is the Python code syntactically valid?",
            "tests": "Does the code pass all provided unit test cases correctly?",
            "documentation": "Does the code include a docstring explaining what the function does?"
        }

    async def grade(self, task_instance: TaskInstance, solution: str) -> tuple[float, dict[str, float], dict[str, str]]:
        ref = task_instance.reference
        code = self._extract_code(solution)
        
        # Rubrics
        syntax_score = 0.0
        test_score = 0.0
        doc_score = 0.0
        
        syntax_fb = "No code found."
        test_fb = "Tests not run."
        doc_fb = "No docstring found."

        if not code:
            return 0.0, {"syntax": 0.0, "tests": 0.0, "documentation": 0.0}, {"syntax": syntax_fb, "tests": test_fb, "documentation": doc_fb}

        # 1. Syntax (weight 0.2)
        try:
            parsed = ast.parse(code)
            syntax_score = 1.0
            syntax_fb = "Syntax is valid."
            
            # Check documentation (weight 0.2)
            has_doc = False
            for node in ast.walk(parsed):
                if isinstance(node, ast.FunctionDef) and ast.get_docstring(node):
                    has_doc = True
                    break
            if has_doc:
                doc_score = 1.0
                doc_fb = "Docstring present."
            else:
                doc_fb = "Missing function docstring."
                
        except SyntaxError as e:
            syntax_fb = f"Syntax error: {e}"
            return 0.0, {"syntax": 0.0, "tests": 0.0, "documentation": 0.0}, {"syntax": syntax_fb, "tests": test_fb, "documentation": doc_fb}

        # 2. Test Cases (weight 0.6)
        passed, total, test_fb = 0, len(ref["test_cases"]), "Tests not run."
        
        def run_tests_sync():
            local_passed = 0
            namespace = {}
            exec(compile(ast.parse(code), '<string>', 'exec'), namespace)
            func = None
            for name, obj in namespace.items():
                if callable(obj) and not name.startswith('_'):
                    func = obj
                    break
            if not func:
                return 0, "No callable function found."
            for input_val, expected in ref["test_cases"]:
                try:
                    if func(input_val) == expected:
                        local_passed += 1
                except Exception:
                    pass
            return local_passed, f"Passed {local_passed}/{total} tests."

        try:
            # Run the highly dangerous eval in a thread with a strict timeout
            passed, test_fb = await asyncio.wait_for(
                asyncio.to_thread(run_tests_sync),
                timeout=2.0
            )
            test_score = passed / total
        except asyncio.TimeoutError:
            test_fb = "Execution timeout: Code took too long (possible infinite loop)."
            test_score = 0.0
        except Exception as e:
            test_fb = f"Execution error: {type(e).__name__}"
            test_score = 0.0

        total_score = 0.2 * syntax_score + 0.6 * test_score + 0.2 * doc_score
        
        rubrics = {"syntax": syntax_score, "tests": test_score, "documentation": doc_score}
        feedback = {"syntax": syntax_fb, "tests": test_fb, "documentation": doc_fb}
        
        return total_score, rubrics, feedback
