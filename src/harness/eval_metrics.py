"""Success metrics for Agentic episode records."""

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from statistics import fmean
from typing import Any


def summarize_episode_success(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Pool episodes overall and give each observed repeat equal weight within its task."""
    statuses: Counter[str] = Counter()
    by_task: dict[str, dict[int, Counter[str]]] = defaultdict(lambda: defaultdict(Counter))
    for record in records:
        status = record["status"]
        statuses[status] += 1
        by_task[record["task"]][record["repeat_id"]][status] += 1

    task_rates = {}
    repeat_rates = {}
    for task, repeats in sorted(by_task.items()):
        task_repeats = {}
        for repeat_id, counts in sorted(repeats.items()):
            episodes = counts.total()
            successes = counts["success"]
            task_repeats[str(repeat_id)] = {
                "episodes": episodes,
                "successes": successes,
                "success_rate": successes / episodes,
            }
        repeat_rates[task] = task_repeats
        task_rates[task] = fmean(result["success_rate"] for result in task_repeats.values())

    episodes = statuses.total()
    successes = statuses["success"]
    return {
        "episodes": episodes,
        "successes": successes,
        "success_rate": successes / episodes if episodes else 0,
        "task_success_rates": task_rates,
        "task_repeat_success_rates": repeat_rates,
        "status_counts": dict(sorted(statuses.items())),
    }
