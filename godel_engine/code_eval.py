from __future__ import annotations

import ast
import base64
import json
import subprocess
import sys
from typing import Any, Sequence


BANNED_CALLS = {
    "compile",
    "delattr",
    "eval",
    "exec",
    "getattr",
    "globals",
    "help",
    "input",
    "locals",
    "open",
    "setattr",
    "vars",
    "__import__",
}

BANNED_NAMES = {
    "os",
    "pathlib",
    "shutil",
    "socket",
    "subprocess",
    "sys",
}

BANNED_NODES = (
    ast.AsyncFunctionDef,
    ast.ClassDef,
    ast.Delete,
    ast.Global,
    ast.Import,
    ast.ImportFrom,
    ast.Nonlocal,
    ast.Raise,
    ast.Try,
    ast.With,
)

HARNESS = r"""
import ast
import base64
import json
import math
import sys

SAFE_BUILTINS = {
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "dict": dict,
    "enumerate": enumerate,
    "float": float,
    "int": int,
    "len": len,
    "list": list,
    "max": max,
    "min": min,
    "pow": pow,
    "range": range,
    "reversed": reversed,
    "round": round,
    "set": set,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "zip": zip,
}

BANNED_CALLS = {
    "compile",
    "delattr",
    "eval",
    "exec",
    "getattr",
    "globals",
    "help",
    "input",
    "locals",
    "open",
    "setattr",
    "vars",
    "__import__",
}

BANNED_NAMES = {"os", "pathlib", "shutil", "socket", "subprocess", "sys"}
BANNED_NODES = (
    ast.AsyncFunctionDef,
    ast.ClassDef,
    ast.Delete,
    ast.Global,
    ast.Import,
    ast.ImportFrom,
    ast.Nonlocal,
    ast.Raise,
    ast.Try,
    ast.With,
)

def validate(tree):
    violations = []
    for node in ast.walk(tree):
        if isinstance(node, BANNED_NODES):
            violations.append(f"Disallowed syntax: {type(node).__name__}")
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            violations.append("Dunder attribute access is not allowed")
        if isinstance(node, ast.Name) and node.id in BANNED_NAMES:
            violations.append(f"Forbidden name: {node.id}")
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in BANNED_CALLS:
                violations.append(f"Forbidden call: {func.id}")
    return violations

payload = json.loads(base64.b64decode(sys.argv[1]).decode("utf-8"))
code = payload["code"]
function_name = payload["function_name"]
test_cases = payload["test_cases"]

try:
    tree = ast.parse(code)
    violations = validate(tree)
    if violations:
        print(json.dumps({"passed": 0, "total": len(test_cases), "error": "; ".join(sorted(set(violations)))}))
        raise SystemExit(0)

    namespace = {"__builtins__": SAFE_BUILTINS, "math": math}
    exec(compile(tree, "<candidate>", "exec"), namespace, namespace)
    func = namespace.get(function_name)
    if not callable(func):
        print(json.dumps({"passed": 0, "total": len(test_cases), "error": f"Function `{function_name}` was not defined."}))
        raise SystemExit(0)

    passed = 0
    failures = []
    for raw_input, expected in test_cases:
        try:
            output = func(raw_input)
        except Exception as exc:
            failures.append(f"{raw_input!r} raised {type(exc).__name__}")
            continue
        if output == expected:
            passed += 1
        else:
            failures.append(f"{raw_input!r} -> {output!r}, expected {expected!r}")

    print(json.dumps({"passed": passed, "total": len(test_cases), "error": "; ".join(failures[:3])}))
except Exception as exc:
    print(json.dumps({"passed": 0, "total": len(test_cases), "error": f"{type(exc).__name__}: {exc}"}))
"""


def extract_code(solution: str) -> str:
    parts = solution.split("```")
    if len(parts) >= 3:
        candidate = parts[1]
        if candidate.startswith("python"):
            candidate = candidate[len("python") :]
        return candidate.strip()
    return solution.strip()


def parse_code_features(code: str) -> dict[str, Any]:
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return {
            "syntax_ok": False,
            "syntax_error": str(exc),
            "tree": None,
            "function_defs": [],
            "has_docstring": False,
            "has_type_hints": False,
        }

    function_defs = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    has_docstring = any(ast.get_docstring(node) for node in function_defs)
    has_type_hints = any(
        node.returns is not None
        or any(arg.annotation is not None for arg in node.args.args)
        for node in function_defs
    )
    return {
        "syntax_ok": True,
        "syntax_error": "",
        "tree": tree,
        "function_defs": function_defs,
        "has_docstring": has_docstring,
        "has_type_hints": has_type_hints,
    }


def validate_code_tree(tree: ast.AST) -> list[str]:
    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, BANNED_NODES):
            violations.append(f"Disallowed syntax: {type(node).__name__}")
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            violations.append("Dunder attribute access is not allowed")
        if isinstance(node, ast.Name) and node.id in BANNED_NAMES:
            violations.append(f"Forbidden name: {node.id}")
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in BANNED_CALLS:
                violations.append(f"Forbidden call: {func.id}")
    return violations


def run_code_tests(
    code: str,
    *,
    function_name: str,
    test_cases: Sequence[tuple[Any, Any]],
    timeout_s: float = 2.0,
) -> dict[str, Any]:
    features = parse_code_features(code)
    if not features["syntax_ok"]:
        return {
            "passed": 0,
            "total": len(test_cases),
            "error": f"Syntax error: {features['syntax_error']}",
        }

    violations = validate_code_tree(features["tree"])
    if violations:
        return {
            "passed": 0,
            "total": len(test_cases),
            "error": "; ".join(sorted(set(violations))),
        }

    payload = {
        "code": code,
        "function_name": function_name,
        "test_cases": list(test_cases),
    }
    encoded = base64.b64encode(json.dumps(payload).encode("utf-8")).decode("ascii")

    try:
        completed = subprocess.run(
            [sys.executable, "-I", "-c", HARNESS, encoded],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {
            "passed": 0,
            "total": len(test_cases),
            "error": f"Execution timed out after {timeout_s:.1f}s.",
        }

    output = completed.stdout.strip()
    if not output:
        return {
            "passed": 0,
            "total": len(test_cases),
            "error": completed.stderr.strip() or "The evaluator returned no output.",
        }

    try:
        result = json.loads(output)
    except json.JSONDecodeError:
        return {
            "passed": 0,
            "total": len(test_cases),
            "error": f"Invalid evaluator output: {output[:160]}",
        }

    return {
        "passed": int(result.get("passed", 0)),
        "total": int(result.get("total", len(test_cases))),
        "error": str(result.get("error", "")),
    }
