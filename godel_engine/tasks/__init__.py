"""
Real-world Task implementations for the Gödel Env.
"""
from godel_engine.tasks.base import BaseTask, TaskInstance
from godel_engine.tasks.alignment_qa import AlignmentQATask
from godel_engine.tasks.python_optimized import PythonOptimizedTask
from godel_engine.tasks.adr_writing import ADRWritingTask
from godel_engine.tasks.strategy_optimization import StrategyOptimizationTask

__all__ = [
    "BaseTask", 
    "TaskInstance", 
    "AlignmentQATask", 
    "PythonOptimizedTask", 
    "ADRWritingTask",
    "StrategyOptimizationTask"
]
