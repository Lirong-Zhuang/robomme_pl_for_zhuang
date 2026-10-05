import json
from pathlib import Path

import h5py
import numpy as np

from mme_vla_suite.dataset_builder.build_reporter_dataset import DatasetBuilder
from mme_vla_suite.reporter_prompts import REPORTER_SYSTEM_PROMPT


def _write_episode(path: Path, *, failure_recovery: bool = False) -> None:
    with h5py.File(path, "w") as data:
        data.attrs["task"] = "BinFill"
        episode = data.create_group("episode_0")
        if failure_recovery:
            episode.attrs["dataset_origin"] = "failure_recovery"
            episode.attrs["failure_recovery"] = True
            episode.attrs["failure_mode"] = "z"
            episode.attrs["difficulty"] = "easy"
        setup = episode.create_group("setup")
        setup.create_dataset("task_goal", data=np.array([b"test task"]))

        for idx in range(101):
            timestep = episode.create_group(f"timestep_{idx}")
            obs = timestep.create_group("obs")
            obs.create_dataset(
                "front_rgb",
                data=np.full((8, 8, 3), idx % 255, dtype=np.uint8),
            )
            if failure_recovery:
                action = timestep.create_group("action")
                ready = np.array([0.1, 0.2, 0.15, 0.0, 0.0, 0.0, 1.0])
                failed = np.array([0.1, 0.2, 0.05, 0.0, 0.0, 0.0, -1.0])
                retry_ready = np.array(
                    [0.11, 0.2, 0.15, 0.0, 0.0, 0.0, 1.0]
                )
                retry_grasp = np.array(
                    [0.11, 0.2, 0.05, 0.0, 0.0, 0.0, -1.0]
                )
                retry_lift = np.array(
                    [0.11, 0.2, 0.15, 0.0, 0.0, 0.0, -1.0]
                )
                if idx < 5:
                    waypoint = ready
                elif idx < 10:
                    waypoint = failed
                elif idx < 15:
                    waypoint = ready
                elif idx < 20:
                    waypoint = retry_ready
                elif idx < 25:
                    waypoint = retry_grasp
                else:
                    waypoint = retry_lift
                action.create_dataset("waypoint_action", data=waypoint)
            info = timestep.create_group("info")
            info.create_dataset("is_video_demo", data=False)
            info.create_dataset("is_completed", data=idx == 100)
            if idx < 40:
                simple = b"first subgoal"
                grounded = b"first subgoal at <10, 20>"
            elif idx < 50:
                simple = b"second subgoal"
                grounded = b"second subgoal at <30, 40>"
            else:
                simple = b"third subgoal"
                grounded = b"third subgoal at <50, 60>"
            info.create_dataset("simple_subgoal", data=simple)
            info.create_dataset("grounded_subgoal", data=grounded)


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_reporter_rows_follow_manager_selection_and_duplication(tmp_path: Path):
    raw_dir = tmp_path / "raw"
    output_dir = tmp_path / "processed"
    raw_dir.mkdir()
    _write_episode(raw_dir / "data_BinFill.h5")

    builder = DatasetBuilder(
        raw_data_path=str(raw_dir),
        preprocessed_data_path=str(output_dir),
    )
    counts = builder.run()

    assert counts == [{"positive": 2, "negative": 3, "duplicates": 1}]
    simple_rows = _read_jsonl(
        output_dir / "reporter_qwenvl" / "simple_subgoal_train.jsonl"
    )
    grounded_rows = _read_jsonl(
        output_dir / "reporter_qwenvl" / "grounded_subgoal_train.jsonl"
    )
    assert len(simple_rows) == len(grounded_rows) == 6

    labels = [json.loads(row["messages"][2]["content"])["success"] for row in simple_rows]
    assert labels == [False, True, True, True, False, False]

    first_row = simple_rows[0]
    assert first_row["messages"][0] == {
        "role": "system",
        "content": REPORTER_SYSTEM_PROMPT,
    }
    assert first_row["messages"][1]["role"] == "user"
    assert "Current Subgoal: first subgoal" in first_row["messages"][1]["content"]
    assert (
        "Recent observation 1/1 (current observation): <image>"
        in first_row["messages"][1]["content"]
    )
    assert "robot's state or pose may be sufficient" in first_row["messages"][1]["content"]
    assert (
        "subgoals involving an object or the environment"
        in first_row["messages"][1]["content"]
    )
    assert first_row["messages"][1]["content"].count("<image>") == 2
    assert first_row["messages"][2] == {
        "role": "assistant",
        "content": '{"success": false}',
    }
    assert len(first_row["images"]) == 2
    assert first_row["images"][0].endswith("step0.png")
    assert first_row["images"][-1].endswith("step20.png")

    # Only selected Reporter-call observations enter the sliding window. The
    # raw HDF5 steps between 20 and 40 are not sampled as adjacent video frames.
    assert len(simple_rows[1]["images"]) == 3
    assert simple_rows[1]["images"][0].endswith("step0.png")
    assert simple_rows[1]["images"][-2].endswith("step20.png")
    assert simple_rows[1]["images"][-1].endswith("step40.png")

    # The transition at step 40 is duplicated exactly as in the Manager data.
    assert simple_rows[1] == simple_rows[2]
    assert grounded_rows[1] == grounded_rows[2]
    # The next Reporter span starts from the newly completed transition frame.
    assert simple_rows[3]["images"][0].endswith("step40.png")
    # Its Reporter-call window was cleared at the subgoal boundary.
    assert len(simple_rows[3]["images"]) == 2
    assert simple_rows[3]["images"][-1].endswith("step50.png")


def test_reporter_can_disable_duplicate_rows_for_test_data(tmp_path: Path):
    raw_dir = tmp_path / "raw"
    output_dir = tmp_path / "processed"
    raw_dir.mkdir()
    _write_episode(raw_dir / "data_BinFill.h5")

    builder = DatasetBuilder(
        raw_data_path=str(raw_dir),
        preprocessed_data_path=str(output_dir),
        duplicate_samples=False,
    )
    counts = builder.run()

    assert counts == [{"positive": 2, "negative": 3, "duplicates": 0}]
    simple_rows = _read_jsonl(
        output_dir / "reporter_qwenvl" / "simple_subgoal_train.jsonl"
    )
    labels = [
        json.loads(row["messages"][2]["content"])["success"]
        for row in simple_rows
    ]
    assert labels == [False, True, True, False, False]
    assert len(simple_rows) == len({json.dumps(row, sort_keys=True) for row in simple_rows})


def test_reporter_history_size_is_configurable(tmp_path: Path):
    raw_dir = tmp_path / "raw"
    output_dir = tmp_path / "processed"
    raw_dir.mkdir()
    _write_episode(raw_dir / "data_BinFill.h5")

    builder = DatasetBuilder(
        raw_data_path=str(raw_dir),
        preprocessed_data_path=str(output_dir),
        duplicate_samples=False,
        reporter_history_size=2,
    )
    builder.run()

    rows = _read_jsonl(
        output_dir / "reporter_qwenvl" / "simple_subgoal_train.jsonl"
    )
    assert [len(row["images"]) for row in rows] == [2, 3, 2, 2, 3]
    assert rows[0]["images"][0].endswith("step0.png")
    assert rows[0]["images"][1].endswith("step20.png")
    assert rows[1]["images"][1].endswith("step20.png")
    assert rows[1]["images"][2].endswith("step40.png")
    assert rows[0]["messages"][1]["content"].count("<image>") == 2
    assert rows[1]["messages"][1]["content"].count("<image>") == 3


def test_failure_recovery_adds_verified_false_hard_negative(tmp_path: Path):
    raw_dir = tmp_path / "raw"
    output_dir = tmp_path / "processed"
    raw_dir.mkdir()
    h5_path = raw_dir / "record_dataset_BinFill_with_failure.h5"
    _write_episode(h5_path, failure_recovery=True)

    builder = DatasetBuilder(
        raw_data_path=str(raw_dir),
        preprocessed_data_path=str(output_dir),
        duplicate_samples=False,
    )
    builder.run()

    rows = _read_jsonl(
        output_dir / "reporter_qwenvl" / "simple_subgoal_train.jsonl"
    )
    hard_negative_rows = [
        row for row in rows if row["images"][-1].endswith("step14.png")
    ]
    assert len(hard_negative_rows) == 1
    assert json.loads(
        hard_negative_rows[0]["messages"][2]["content"]
    ) == {"success": False}

    audit = json.loads(
        (
            output_dir
            / "reporter_qwenvl"
            / "failure_hard_negative_audit.json"
        ).read_text()
    )
    assert audit["failure_recovery_episode_count"] == 1
    assert audit["failure_hard_negative_count"] == 1
    assert audit["all_labels_are_false"] is True
    assert audit["records"][0]["episode"] == 0
    assert audit["records"][0]["step"] == 14
    assert audit["records"][0]["success"] is False
