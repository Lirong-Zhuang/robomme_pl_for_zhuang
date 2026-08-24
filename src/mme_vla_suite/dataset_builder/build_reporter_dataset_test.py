import json
from pathlib import Path

import h5py
import numpy as np

from mme_vla_suite.dataset_builder.build_reporter_dataset import DatasetBuilder
from mme_vla_suite.reporter_prompts import REPORTER_SYSTEM_PROMPT


def _write_episode(path: Path) -> None:
    with h5py.File(path, "w") as data:
        episode = data.create_group("episode_0")
        setup = episode.create_group("setup")
        setup.create_dataset("task_goal", data=np.array([b"test task"]))

        for idx in range(101):
            timestep = episode.create_group(f"timestep_{idx}")
            obs = timestep.create_group("obs")
            obs.create_dataset(
                "front_rgb",
                data=np.full((8, 8, 3), idx % 255, dtype=np.uint8),
            )
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
    assert "Observation after execution: <image>" in first_row["messages"][1]["content"]
    assert "The episode may contain many subgoals" in first_row["messages"][1]["content"]
    assert first_row["messages"][2] == {
        "role": "assistant",
        "content": '{"success": false}',
    }
    assert first_row["images"][0].endswith("step0.png")
    assert first_row["images"][1].endswith("step20.png")

    # The transition at step 40 is duplicated exactly as in the Manager data.
    assert simple_rows[1] == simple_rows[2]
    assert grounded_rows[1] == grounded_rows[2]
    # The next Reporter span starts from the newly completed transition frame.
    assert simple_rows[3]["images"][0].endswith("step40.png")
    assert simple_rows[3]["images"][1].endswith("step50.png")
