from harness.eval_metrics import summarize_episode_success


def test_success_metrics_include_each_task_and_repeat() -> None:
    metrics = summarize_episode_success([
        {"task": "BinFill", "repeat_id": 2, "status": "success"},
        {"task": "SwingXtimes", "repeat_id": 1, "status": "fail"},
        {"task": "BinFill", "repeat_id": 1, "status": "success"},
        {"task": "BinFill", "repeat_id": 2, "status": "timeout"},
        {"task": "BinFill", "repeat_id": 2, "status": "error"},
        {"task": "BinFill", "repeat_id": 2, "status": "fail"},
    ])

    assert metrics["episodes"] == 6
    assert metrics["successes"] == 2
    assert metrics["success_rate"] == 2 / 6
    assert metrics["task_success_rates"] == {"BinFill": 0.625, "SwingXtimes": 0.0}
    assert metrics["task_repeat_success_rates"] == {
        "BinFill": {
            "1": {"episodes": 1, "successes": 1, "success_rate": 1.0},
            "2": {"episodes": 4, "successes": 1, "success_rate": 0.25},
        },
        "SwingXtimes": {"1": {"episodes": 1, "successes": 0, "success_rate": 0.0}},
    }
    assert metrics["status_counts"] == {"error": 1, "fail": 2, "success": 2, "timeout": 1}


def test_empty_summary_preserves_zero_overall_rate() -> None:
    assert summarize_episode_success([]) == {
        "episodes": 0,
        "successes": 0,
        "success_rate": 0,
        "task_success_rates": {},
        "task_repeat_success_rates": {},
        "status_counts": {},
    }
