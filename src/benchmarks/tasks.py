"""The lm-eval TaskManager every benchmark path shares (src/benchmarks/runner.py,
src/steering/task_data.py): lm-eval's built-in tasks plus this repo's own
task definitions in configs/benchmarks/lm_eval_tasks/ (sciq_closed,
mmlu_cont), so configs/benchmarks/tasks.yaml can list either kind.
"""

import os

from lm_eval.tasks import TaskManager

TASKS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                         "configs", "benchmarks", "lm_eval_tasks")

_TASK_MANAGER = None


def task_manager():
    global _TASK_MANAGER
    if _TASK_MANAGER is None:
        _TASK_MANAGER = TaskManager(include_path=TASKS_DIR)
    return _TASK_MANAGER
