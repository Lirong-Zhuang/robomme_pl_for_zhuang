"""Merge original and failure-recovery RoboMME episodes into one HDF5 file.

This is a lossless packaging step. It does not create train/test splits and it
does not relabel Reporter samples. Original episode IDs are retained. Recovery
episodes are remapped after the final original ID so the two sources cannot
collide. Generation metadata from the JSON manifests is attached to each
copied recovery group as HDF5 attributes.

Server example, run from the repository root::

    uv run --project third_party/robomme_benchmark \
      python scripts/merge_episode_h5.py \
      --base-h5 /data/public/RoboMME/record_dataset_BinFill.h5 \
      --base-expected-count 100 \
      --input-dir data/h5_data/hdf5_files \
      --manifest-dir data/h5_data \
      --task BinFill \
      --expected-count 30 \
      --expected-start-episode 0 \
      --output data/h5_data/merged/record_dataset_BinFill_with_failure.h5
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any

import h5py


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge per-episode RoboMME HDF5 files without creating a "
            "train/test split."
        )
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument(
        "--base-h5",
        type=Path,
        default=None,
        help=(
            "Optional original task-level HDF5 to copy before recovery episodes. "
            "Its episode IDs are retained."
        ),
    )
    parser.add_argument(
        "--base-expected-count",
        type=int,
        default=None,
        help="Require this many episodes in --base-h5.",
    )
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=None,
        help="Directory containing generation_manifest_*.json (default: input parent).",
    )
    parser.add_argument("--task", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=None)
    parser.add_argument("--expected-start-episode", type=int, default=0)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Atomically replace an existing merged output.",
    )
    args = parser.parse_args(argv)
    if args.expected_count is not None and args.expected_count < 1:
        parser.error("--expected-count must be at least 1")
    if args.base_expected_count is not None and args.base_expected_count < 1:
        parser.error("--base-expected-count must be at least 1")
    if args.base_expected_count is not None and args.base_h5 is None:
        parser.error("--base-expected-count requires --base-h5")
    if args.expected_start_episode < 0:
        parser.error("--expected-start-episode must be non-negative")
    return args


def _episode_sources(
    input_dir: Path,
    task: str,
) -> list[tuple[int, int, Path]]:
    pattern = re.compile(
        rf"^{re.escape(task)}_ep(?P<episode>\d+)_seed(?P<seed>\d+)\.h5$"
    )
    sources: list[tuple[int, int, Path]] = []
    ignored_h5: list[str] = []
    for path in sorted(input_dir.glob("*.h5")):
        match = pattern.match(path.name)
        if match is None:
            ignored_h5.append(path.name)
            continue
        sources.append(
            (int(match.group("episode")), int(match.group("seed")), path)
        )

    if ignored_h5:
        print("Ignoring HDF5 files that do not match this task:")
        for name in ignored_h5:
            print(f"  {name}")
    if not sources:
        raise RuntimeError(
            f"No {task}_ep*_seed*.h5 files found in {input_dir}"
        )

    episodes = [episode for episode, _, _ in sources]
    duplicate_episodes = sorted(
        episode for episode in set(episodes) if episodes.count(episode) > 1
    )
    if duplicate_episodes:
        raise RuntimeError(
            f"Multiple source files found for episodes: {duplicate_episodes}"
        )
    return sorted(sources)


def _load_generation_metadata(
    manifest_dir: Path,
    task: str,
) -> dict[int, dict[str, Any]]:
    records: dict[int, dict[str, Any]] = {}
    manifest_paths = sorted(manifest_dir.glob("generation_manifest_*.json"))
    if not manifest_paths:
        raise RuntimeError(
            f"No generation_manifest_*.json files found in {manifest_dir}"
        )

    for manifest_path in manifest_paths:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("task") != task:
            continue
        for raw_record in manifest.get("episodes", []):
            record = dict(raw_record)
            episode = int(record["episode"])
            if episode in records:
                raise RuntimeError(
                    f"Episode {episode} occurs in multiple manifests, including "
                    f"{manifest_path}"
                )
            record["manifest_file"] = manifest_path.name
            records[episode] = record

    if not records:
        raise RuntimeError(
            f"No {task} episode records found in manifests under {manifest_dir}"
        )
    return records


def _validate_expected_episodes(
    sources: list[tuple[int, int, Path]],
    expected_count: int | None,
    expected_start: int,
) -> None:
    if expected_count is None:
        return
    actual = {episode for episode, _, _ in sources}
    expected = set(range(expected_start, expected_start + expected_count))
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if missing or unexpected:
        raise RuntimeError(
            f"Episode coverage mismatch; missing={missing}, unexpected={unexpected}"
        )


def _validate_source(
    source: h5py.File,
    *,
    episode: int,
    seed: int,
    path: Path,
) -> str:
    episode_key = f"episode_{episode}"
    episode_keys = sorted(key for key in source if key.startswith("episode_"))
    if episode_keys != [episode_key]:
        raise RuntimeError(
            f"{path} contains episode groups {episode_keys}, expected [{episode_key!r}]"
        )
    episode_group = source[episode_key]
    timestep_count = sum(
        key.startswith("timestep_") for key in episode_group.keys()
    )
    if timestep_count == 0:
        raise RuntimeError(f"{path} contains no timesteps")
    if "setup/seed" not in episode_group:
        raise RuntimeError(f"{path} is missing {episode_key}/setup/seed")
    stored_seed = int(episode_group["setup/seed"][()])
    if stored_seed != seed:
        raise RuntimeError(
            f"{path} filename seed={seed}, but setup/seed={stored_seed}"
        )
    return episode_key


def _attach_generation_metadata(
    episode_group: h5py.Group,
    record: dict[str, Any],
    source_path: Path,
) -> None:
    episode_group.attrs["dataset_origin"] = "failure_recovery"
    episode_group.attrs["failure_recovery"] = True
    episode_group.attrs["failure_mode"] = str(record["failure_mode"])
    episode_group.attrs["difficulty"] = str(record["difficulty"])
    episode_group.attrs["generation_attempts"] = int(record["attempts"])
    episode_group.attrs["generation_seed"] = int(record["seed"])
    episode_group.attrs["source_h5"] = source_path.name
    episode_group.attrs["generation_manifest"] = str(record["manifest_file"])


def _episode_keys(data: h5py.File) -> list[tuple[int, str]]:
    keys: list[tuple[int, str]] = []
    for key in data.keys():
        match = re.fullmatch(r"episode_(\d+)", key)
        if match is not None:
            keys.append((int(match.group(1)), key))
    return sorted(keys)


def _copy_base_episodes(
    base_h5: Path,
    destination: h5py.File,
    expected_count: int | None,
) -> list[int]:
    if not base_h5.is_file():
        raise FileNotFoundError(f"Base HDF5 does not exist: {base_h5}")

    with h5py.File(base_h5, "r") as source:
        episode_keys = _episode_keys(source)
        if not episode_keys:
            raise RuntimeError(f"Base HDF5 contains no episode groups: {base_h5}")
        if expected_count is not None and len(episode_keys) != expected_count:
            raise RuntimeError(
                f"Base HDF5 contains {len(episode_keys)} episodes, "
                f"expected {expected_count}: {base_h5}"
            )

        episode_ids = [episode for episode, _ in episode_keys]
        expected_ids = list(range(episode_ids[0], episode_ids[0] + len(episode_ids)))
        if episode_ids != expected_ids:
            raise RuntimeError(
                f"Base HDF5 episode IDs are not contiguous: {episode_ids}"
            )

        for episode, episode_key in episode_keys:
            group = source[episode_key]
            timestep_count = sum(
                key.startswith("timestep_") for key in group.keys()
            )
            if timestep_count == 0:
                raise RuntimeError(
                    f"Base HDF5 {episode_key} contains no timesteps: {base_h5}"
                )
            source.copy(group, destination, name=episode_key)
            copied = destination[episode_key]
            copied.attrs["dataset_origin"] = "original"
            copied.attrs["source_episode"] = episode
            copied.attrs["source_h5"] = base_h5.name
            print(f"copied base {episode_key}")

    return episode_ids


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def merge(args: argparse.Namespace) -> Path:
    input_dir = args.input_dir.expanduser().resolve()
    manifest_dir = (
        args.manifest_dir.expanduser().resolve()
        if args.manifest_dir is not None
        else input_dir.parent
    )
    output = args.output.expanduser().resolve()
    base_h5 = (
        args.base_h5.expanduser().resolve()
        if args.base_h5 is not None
        else None
    )

    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    if output.exists() and not args.overwrite:
        raise FileExistsError(
            f"Refusing to replace existing output: {output}. Pass --overwrite "
            "only after checking the existing file."
        )

    sources = _episode_sources(input_dir, args.task)
    _validate_expected_episodes(
        sources,
        args.expected_count,
        args.expected_start_episode,
    )
    metadata = _load_generation_metadata(manifest_dir, args.task)

    source_episodes = {episode for episode, _, _ in sources}
    metadata_episodes = set(metadata)
    if source_episodes != metadata_episodes:
        raise RuntimeError(
            "Manifest/source mismatch; "
            f"missing metadata={sorted(source_episodes - metadata_episodes)}, "
            f"metadata without HDF5={sorted(metadata_episodes - source_episodes)}"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.",
        suffix=".tmp",
        dir=output.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)

    try:
        with h5py.File(temporary, "w") as destination:
            destination.attrs["task"] = args.task
            destination.attrs["dataset_kind"] = (
                "original_plus_failure_recovery"
                if base_h5 is not None
                else "failure_recovery"
            )

            base_episode_ids: list[int] = []
            if base_h5 is not None:
                base_episode_ids = _copy_base_episodes(
                    base_h5,
                    destination,
                    args.base_expected_count,
                )
                destination.attrs["base_h5"] = str(base_h5)

            recovery_output_start = (
                max(base_episode_ids) + 1 if base_episode_ids else min(source_episodes)
            )
            source_episode_start = min(source_episodes)

            for episode, seed, source_path in sources:
                record = metadata[episode]
                manifest_seed = int(record["seed"])
                manifest_h5_name = Path(str(record["h5_path"])).name
                if manifest_seed != seed:
                    raise RuntimeError(
                        f"Episode {episode}: filename seed={seed}, "
                        f"manifest seed={manifest_seed}"
                    )
                if manifest_h5_name != source_path.name:
                    raise RuntimeError(
                        f"Episode {episode}: source file is {source_path.name}, "
                        f"manifest points to {manifest_h5_name}"
                    )

                with h5py.File(source_path, "r") as source:
                    source_episode_key = _validate_source(
                        source,
                        episode=episode,
                        seed=seed,
                        path=source_path,
                    )
                    output_episode = (
                        recovery_output_start + episode - source_episode_start
                    )
                    output_episode_key = f"episode_{output_episode}"
                    if output_episode_key in destination:
                        raise RuntimeError(
                            f"Output episode collision: {output_episode_key}"
                        )
                    source.copy(
                        source[source_episode_key],
                        destination,
                        name=output_episode_key,
                    )

                copied_group = destination[output_episode_key]
                _attach_generation_metadata(copied_group, record, source_path)
                copied_group.attrs["source_episode"] = episode
                print(
                    f"copied {source_path.name} -> {output_episode_key} "
                    f"mode={record['failure_mode']} difficulty={record['difficulty']}"
                )

            destination.attrs["original_episode_count"] = len(base_episode_ids)
            destination.attrs["failure_recovery_episode_count"] = len(sources)
            destination.attrs["episode_count"] = len(base_episode_ids) + len(sources)
            destination.flush()

        os.replace(temporary, output)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise

    print(
        f"Merged {len(base_episode_ids)} original + {len(sources)} "
        f"failure-recovery episodes: {output}"
    )
    print(f"SHA256: {_sha256(output)}")
    return output


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    merge(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
