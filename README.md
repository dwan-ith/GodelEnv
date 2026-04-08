---
title: GodelEnv
emoji: "🚞"
colorFrom: purple
colorTo: blue
sdk: docker
pinned: false
tags:
  - openenv
  - reinforcement-learning
---
# ♾️ Gödel Env

> A self-improving reinforcement learning environment built on [OpenEnv](https://openenv.dev) inspired by Gödel Machines.

[![OpenEnv](https://img.shields.io/badge/OpenEnv-compatible-blue)](https://openenv.dev)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An LLM agent iteratively improves solutions across professional real-world domains. Each step is graded by a multi-axis **LLM-as-a-judge** rubric grader, producing partial progress signals (not sparse binary rewards). The hardest task ("Gödel tier") is fully recursive: the agent must improve a *reasoning template* that is then empirically tested on a downstream challenge.

---

## Architecture

```
godel_engine/           ← Core RL package (no web server needed)
│   environment.py      ← GodelEnvironment: reset() / step() / state()
│   agent.py            ← AutoAgent: LLM-based action policy
│   evolution.py        ← DarwinPool + HuxleyTracker (strategy evolution)
│   models.py           ← Typed Pydantic models (GodelAction, GodelObservation…)
│   graders/
│       agent_grader.py ← LLM-as-a-judge grader (LiteLLM)
│   tasks/
│       base.py         ← BaseTask abstract class
│       factual_qa.py   ← Easy: factual explanation
│       alignment_qa.py ← Easy: AI safety concepts
│       code_improvement.py ← Medium: Python correctness (deterministic)
│       python_optimized.py ← Medium: code performance + docs
│       reasoning.py    ← Medium: multi-step logic
│       adr_writing.py  ← Hard: Architecture Decision Records
│       strategy_optimization.py ← Gödel: recursive self-improvement
server/                 ← Optional FastAPI dashboard wrapper
baseline.py             ← Standalone RL evaluation script
```

---

## Tasks (easy → medium → hard → gödel)

| Task | Difficulty | Grader | Rubrics |
|---|---|---|---|
| `factual_qa` | Easy | LLM judge | coverage, detail, structure |
| `alignment_qa` | Easy | LLM judge | technical_accuracy, clarity, completeness |
| `code_improvement` | Medium | Deterministic (AST + exec) | syntax, tests, documentation |
| `python_optimized` | Medium | LLM judge | correctness, efficiency, documentation |
| `reasoning` | Medium | LLM judge | logical_validity, completeness, clarity |
| `adr_writing` | Hard | LLM judge | structure, completeness, trade_off_analysis |
| `strategy_optimization` | Gödel | Two-phase (structural + empirical) | self_verification, structural_rigor, recursive_potential, downstream_quality |

---

## Action & Observation Spaces

### Action (`GodelAction`)
```python
class GodelAction(BaseModel):
    solution: str        # Full replacement solution text
    edit_type: EditType  # rewrite | refine | add_reasoning | ...
    strategy_note: str   # Optional agent rationale (logged, not graded)
```

### Observation (`GodelObservation`)
```python
class GodelObservation(BaseModel):
    task_prompt: str              # The problem to solve
    current_solution: str         # Agent's current solution
    total_score: float            # Weighted composite [0.0, 1.0]
    rubric_scores: RubricScores   # Per-axis scores + feedback
    step: int                     # Current step number
    improvement_history: list     # Delta history
    feedback_summary: str         # Plain-text improvement hint
```

### Reward Function
```
reward = (score_t - score_{t-1}) - 0.005 * step_cost
```
Episodes terminate when `score >= 0.95`, the agent stagnates (3 flat steps), or `step >= 10`.

---

##  Quick Start

### 1. Install

```bash
git clone <repo-url>
cd GodelEngine
pip install -e ".[dev]"
```

### 2. Configure API Key

```bash
cp .env.example .env
# Edit .env and add your key:
# OPENROUTER_API_KEY=sk-or-...   (recommended — free tier available)
# OPENAI_API_KEY=sk-...
# GEMINI_API_KEY=...
# ANTHROPIC_API_KEY=sk-ant-...
```

### 3. Run Baseline (standalone — no server needed)

```bash
# Run all 7 tasks
python baseline.py

# Run specific tasks with seed for reproducibility
python baseline.py --tasks factual_qa code_improvement reasoning --seed 42

# Save results to JSON
python baseline.py --episodes 7 --seed 42 --output results.json
```

Expected output:
```
============================================================
  GÖDEL ENV — BASELINE EVALUATION
============================================================
  Tasks:    ['factual_qa', 'alignment_qa', ...]
  Episodes: 7

── Episode 1/7 | factual_qa ──
  Initial Score: 0.050
  step 01 | [████████████░░░░░░░░░░░░░░░░░░] 0.420 | reward=+0.3700 | ...
  step 02 | [█████████████████████░░░░░░░░░] 0.720 | reward=+0.3000 | ...
  ✓ Done | final=0.720 Δ=+0.670 | steps=2
...
============================================================
  RESULTS SUMMARY
  Avg init score:  0.0621
  Avg final score: 0.7340
  Avg Δ score:    +0.6719
============================================================
```

### 4. Run Dashboard (optional)

```bash
uvicorn server.app:app --host 0.0.0.0 --port 8000
# Open: http://localhost:8000/dashboard/
```

### 5. Use as a Python Library

```python
import asyncio
from godel_engine import GodelEnvironment, AutoAgent, GodelAction

async def main():
    env = GodelEnvironment(seed=42)
    agent = AutoAgent()

    result = await env.reset(task_type="code_improvement")
    print(f"Task: {result.observation.task_prompt}")
    print(f"Baseline score: {result.observation.total_score:.3f}")

    while not result.terminated and not result.truncated:
        action = await agent.act(
            task_prompt=result.observation.task_prompt,
            current_solution=result.observation.current_solution,
            rubrics=result.observation.rubric_scores.scores,
            task_type=result.observation.task_type,
        )
        result = await env.step(action)
        print(f"Step {result.observation.step}: score={result.observation.total_score:.3f}")

asyncio.run(main())
```

---

##  Docker / HF Spaces

```bash
docker build -t godel-env .
docker run -p 7860:7860 -e OPENROUTER_API_KEY=sk-or-... godel-env
```

The dashboard will be live at `http://localhost:7860/dashboard/`.

---

## 🧬 Evolutionary Layers

- **Darwin Pool**: Tournament-based strategy selection. Strategies that produce higher scores survive and spawn children.
- **Huxley Tracker**: Measures **Clade-Metaproductivity (CMP)** — strategies that produce *better descendants* are prioritized, not just strategies with high personal scores.

---

##  Reproducible Baseline

```bash
python baseline.py --episodes 7 --seed 42 --output baseline_results.json
```

| Task | Difficulty | Avg Score | Avg Δ |
|---|---|---|---|
| factual_qa | Easy | ~0.72 | +0.67 |
| alignment_qa | Easy | ~0.68 | +0.63 |
| code_improvement | Medium | ~0.80 | +0.60 |
| python_optimized | Medium | ~0.65 | +0.55 |
| reasoning | Medium | ~0.63 | +0.53 |
| adr_writing | Hard | ~0.58 | +0.48 |
| strategy_optimization | Gödel | ~0.71 | +0.63 |

*(Scores vary with model)*


