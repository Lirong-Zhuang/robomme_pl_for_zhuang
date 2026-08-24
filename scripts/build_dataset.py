"""Single entrypoint to run Executer, Manager, and Reporter dataset builders.

Build Executer preprocessed pickle data from raw HDF5 data.
```
uv run python scripts/build_dataset.py --dataset_type executer
```

Build only selected Manager tasks into a custom output directory.
```
uv run python scripts/build_dataset.py \
  --dataset_type manager_qwenvl \
  --raw_data_path data/robomme_data_h5 \
  --preprocessed_data_path data/trinity_preprocessed_data/manager_binfill_data_1 \
  --tasks BinFill
```

Build Manager subgoal prediction dataset for QwenVL.
```
uv run python scripts/build_dataset.py --dataset_type manager_qwenvl
```

Build Manager subgoal prediction dataset for MemER.
```
uv run python scripts/build_dataset.py --dataset_type manager_memer
```

Build Manager subgoal prediction dataset for QwenVL plus key states.
```
uv run python scripts/build_dataset.py --dataset_type manager_qpa
```

Build Reporter completion-classification data for QwenVL.
```
uv run python scripts/build_dataset.py \
  --dataset_type reporter_qwenvl \
  --raw_data_path data/robomme_data_h5 \
  --preprocessed_data_path data/trinity_preprocessed_data/reporter_binfill_data_1 \
  --tasks BinFill
```

"""

import argparse
import time

from mme_vla_suite.dataset_builder.build_executer_dataset import (
    ExecuterDatasetProcessor,
)
from mme_vla_suite.dataset_builder.build_manager_dataset_memer import (
    DatasetBuilder as MemerManagerDatasetBuilder,
)
from mme_vla_suite.dataset_builder.build_manager_dataset_qpa import (
    QPA_DIR_NAME,
    DatasetBuilder as QPAManagerDatasetBuilder,
)
from mme_vla_suite.dataset_builder.build_manager_dataset_qwenvl import (
    DatasetBuilder as QwenVLManagerDatasetBuilder,
)
from mme_vla_suite.dataset_builder.build_reporter_dataset import (
    DatasetBuilder as ReporterDatasetBuilder,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess raw HDF5 dataset for training"
    )
    parser.add_argument(
        "--dataset_type",
        type=str,
        default="executer",
        choices=[
            "executer",
            "manager_qwenvl",
            "manager_qpa",
            "manager_memer",
            "reporter_qwenvl",
        ],
        help="Dataset type to build",
    )
    parser.add_argument(
        "--raw_data_path",
        type=str,
        default="data/robomme_data_h5",
        help="Raw HDF5 directory",
    )
    parser.add_argument(
        "--preprocessed_data_path",
        type=str,
        default="data/binfill_test_preprocessed",
        help="Output directory",
    )
    parser.add_argument(
        "--max_episodes",
        type=int,
        default=None,
        help="Cap episodes per file (default: all)",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=None,
        help="Only build these RoboMME task names, e.g. --tasks BinFill PickXtimes",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Write visualization MP4s",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    t0 = time.perf_counter()

    if args.dataset_type == "executer":
        processor = ExecuterDatasetProcessor(
            raw_data_path=args.raw_data_path,
            preprocessed_data_path=args.preprocessed_data_path,
            visualize=args.visualize,
            max_episodes=args.max_episodes,
            task_names=args.tasks,
        )
        processor.run()
    elif args.dataset_type == "manager_qwenvl":
        builder = QwenVLManagerDatasetBuilder(
            raw_data_path=args.raw_data_path,
            preprocessed_data_path=args.preprocessed_data_path,
            max_episodes=args.max_episodes,
            visualize=args.visualize,
            manager_dir_name="qwenvl",
            task_names=args.tasks,
        )
        builder.run()
    elif args.dataset_type == "manager_memer":
        builder = MemerManagerDatasetBuilder(
            raw_data_path=args.raw_data_path,
            preprocessed_data_path=args.preprocessed_data_path,
            max_episodes=args.max_episodes,
            visualize=args.visualize,
            manager_dir_name="memer",
            task_names=args.tasks,
        )
        builder.run()
    elif args.dataset_type == "manager_qpa":
        builder = QPAManagerDatasetBuilder(
            raw_data_path=args.raw_data_path,
            preprocessed_data_path=args.preprocessed_data_path,
            max_episodes=args.max_episodes,
            visualize=args.visualize,
            manager_dir_name=QPA_DIR_NAME,
            task_names=args.tasks,
        )
        builder.run()
    elif args.dataset_type == "reporter_qwenvl":
        builder = ReporterDatasetBuilder(
            raw_data_path=args.raw_data_path,
            preprocessed_data_path=args.preprocessed_data_path,
            max_episodes=args.max_episodes,
            visualize=args.visualize,
            reporter_dir_name="reporter_qwenvl",
            task_names=args.tasks,
        )
        builder.run()
    else:
        raise ValueError(f"Unknown dataset_type: {args.dataset_type}")

    print(f"Time taken: {(time.perf_counter() - t0) / 60:.2f} minutes")
