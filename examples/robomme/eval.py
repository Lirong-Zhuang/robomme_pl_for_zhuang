import csv
import dataclasses
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import traceback
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Optional, TextIO, Tuple

# Use the pre-downloaded shared Qwen model and keep Hub access offline by
# default. Set either variable explicitly to override this on another server.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from tqdm import tqdm
from mme_vla_suite.prompts import DEFAULT_MANAGER_PROMPT_VERSION
from mme_vla_suite.prompts import DEFAULT_REPORTER_PROMPT_VERSION
from mme_vla_suite.prompts import get_manager_prompt
from mme_vla_suite.prompts import get_reporter_prompt
from mme_vla_suite.qwen_backbone import DEFAULT_QWEN_BACKBONE_NAME
from mme_vla_suite.qwen_backbone import DEFAULT_QWEN_BACKBONE_ROOT
from mme_vla_suite.qwen_backbone import resolve_qwen_backbone_path
from mme_vla_suite.reporter_gripper_tool import DEFAULT_GRIPPER_CLOSED_POSITION
from mme_vla_suite.reporter_gripper_tool import DEFAULT_GRIPPER_ENDPOINT_BUFFER
from mme_vla_suite.reporter_gripper_tool import DEFAULT_GRIPPER_OPEN_POSITION
from mme_vla_suite.reporter_prompts import DEFAULT_REPORTER_HISTORY_SIZE

# Do not retain unused CUDA allocations on the shared evaluation GPU.
os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"

from utils import (
    check_args,
    TASK_NAME_LIST,
    TASK_WITH_VIDEO_DEMO,
    SUBGOAL_TYPES,
    EpisodeState,
)
from utils import RolloutRecorder
from env_runner import EnvRunner
from manager import build_manager, ManagerBase
from executer import build_executer, Executer
from reporter import build_reporter, ReporterBase

# qwen3-vl environment variables
os.environ['IMAGE_MAX_TOKEN_NUM'] = '256'
os.environ['VIDEO_MAX_TOKEN_NUM'] = '64'
os.environ['FPS_MAX_FRAMES'] = '10'


TIMING_COMPONENTS = ("framework", "reporter", "manager", "executer")
TIMING_UNIT = "ms"
TIMING_DEFINITION = (
    "End-to-end wall-clock latency for one action-chunk inference cycle. "
    "Each timing is calculated as result-ready timestamp minus the timestamp "
    "when that scope receives the current observation. Framework ends when "
    "the action chunk is returned; Reporter, Manager, and Executer end when "
    "their respective results are returned."
)


@dataclasses.dataclass
class Args:
    executer_host: str = "0.0.0.0"
    executer_port: int = 8011

    obs_horizon: int = 16
    max_steps: int = 1300
    save_dir: str = "runs/evaluation"
    run_name: str = ""
    overwrite: bool = True
    save_manager_logs: bool = True

    executer_use_history: bool = True
    executer_name: str = "symbolic-grounded-subgoal"
    executer_seed: int = 7
    executer_ckpt_id: int = 79999
    repeat_id: int = 1
    eval_seeds: str = ""
    num_repeats: int = 1
    # Physical GPU selectors used by the launcher. CUDA remaps each selected
    # device to cuda:0 inside its process, so record the launcher values too.
    executer_gpu_id: str = ""
    manager_reporter_gpu_id: str = os.environ.get("CUDA_VISIBLE_DEVICES", "all")

    # task control
    re_eval_tasks: str = "" # tasks split by comma
    only_tasks: str = "BinFill" # tasks split by comma
    exclude_tasks: str = "" # tasks split by comma

    # Manager
    manager_use_oracle: bool = False
    manager_use_qwenvl: bool = True
    manager_use_memer: bool = False
    manager_use_gemini: bool = False
    # subgoal_type: Optional[str] = "simple_subgoal"  # [simple_subgoal, grounded_subgoal]
    subgoal_type: Optional[str] = "grounded_subgoal"
    manager_gemini_model_name: str = "gemini-2.5-pro"
    manager_simple_adapter_path: str = ""
    manager_grounded_adapter_path: str = ""
    # Applies to the QwenVL Manager only.
    manager_prompt_version: str = DEFAULT_MANAGER_PROMPT_VERSION
    manager_save_memer_kf: bool = False
    subgoal_keep_period: int = 1 # ever subgoal should be kept for this many steps
    manager_qwen_backbone_name: str = os.environ.get(
        "MANAGER_QWEN_BACKBONE_NAME", DEFAULT_QWEN_BACKBONE_NAME
    )
    manager_qwen_backbone_root: str = os.environ.get(
        "MANAGER_QWEN_BACKBONE_ROOT", DEFAULT_QWEN_BACKBONE_ROOT
    )
    manager_qwen_backbone_path: str = os.environ.get(
        "MANAGER_QWEN_BACKBONE_PATH", ""
    )

    # Reporter
    reporter_type: str = "qwenvl"
    reporter_qwen_backbone_name: str = os.environ.get(
        "REPORTER_QWEN_BACKBONE_NAME", DEFAULT_QWEN_BACKBONE_NAME
    )
    reporter_qwen_backbone_root: str = os.environ.get(
        "REPORTER_QWEN_BACKBONE_ROOT", DEFAULT_QWEN_BACKBONE_ROOT
    )
    reporter_qwen_backbone_path: str = os.environ.get(
        "REPORTER_QWEN_BACKBONE_PATH", ""
    )
    reporter_adapter_path: str = ""
    reporter_prompt_version: str = DEFAULT_REPORTER_PROMPT_VERSION
    # Number of recent Reporter-call observations, including the current one.
    # Input is two init images plus up to three overall/wrist observation pairs.
    reporter_history_size: int = DEFAULT_REPORTER_HISTORY_SIZE
    # True: debounce continuous true runs (effective on calls 1, 4, 7, ...).
    # False: match dev_trinity and pass every parsed Reporter result directly.
    reporter_debounce: bool = True
    # Optional deterministic veto for Reporter true on subgoals containing
    # "pick up". Only an observed gripper opening strictly outside the fully
    # closed/open endpoint buffers is accepted as a completed grasp.
    reporter_gripper_gate: bool = False
    reporter_gripper_closed_position: float = DEFAULT_GRIPPER_CLOSED_POSITION
    reporter_gripper_open_position: float = DEFAULT_GRIPPER_OPEN_POSITION
    reporter_gripper_endpoint_buffer: float = DEFAULT_GRIPPER_ENDPOINT_BUFFER
    # this can accelerate the evaluation process for symbolic memory
    # In our experiments, we just set this to 1
    num_episodes: int = 10 # number of episodes to evaluate for each task
    episode_ids: str = "2" # exact episode IDs to evaluate, e.g. "7" or "2,7"; overrides num_episodes


class LogStream:
    """Write captured episode output to its log without echoing to terminal."""

    def __init__(self, log_file: TextIO):
        self.log_file = log_file

    def write(self, text: str) -> int:
        self.log_file.write(text)
        return len(text)

    def flush(self) -> None:
        self.log_file.flush()


@contextmanager
def manager_log(save_dir: Path, task_name: str, episode_id: int, enabled: bool):
    """Capture episode output in the Manager log and keep it off the terminal."""
    if not enabled:
        with open(os.devnull, "w", encoding="utf-8") as null_stream:
            with redirect_stdout(null_stream), redirect_stderr(null_stream):
                yield None
        return

    log_dir = save_dir / task_name / "manager_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{task_name}_ep{episode_id}.log"
    with log_path.open("w", encoding="utf-8", buffering=1) as log_file:
        log_stream = LogStream(log_file)
        with redirect_stdout(log_stream), redirect_stderr(log_stream):
            print(f"[robomme] Manager log: {log_path}")
            yield log_path


def summarize_episode_timing(cycles: list[dict]) -> dict:
    """Return per-component averages without counting skipped component calls."""
    summary = {}
    for component in TIMING_COMPONENTS:
        key = f"{component}_ms"
        values = [
            float(cycle[key])
            for cycle in cycles
            if cycle.get(key) is not None
        ]
        total_ms = sum(values)
        summary[component] = {
            "call_count": len(values),
            "average_ms": (
                round(total_ms / len(values), 3) if values else None
            ),
            "total_ms": round(total_ms, 3),
        }
    return summary


def load_episode_timing_summaries(save_dir: Path, log_dict: dict) -> dict:
    """Load completed per-episode timing summaries for the repeat log."""
    episodes = {}
    for task_name, episode_results in sorted(log_dict.items()):
        task_timings = {}
        for episode_id in sorted(episode_results, key=lambda value: int(value)):
            timing_path = (
                save_dir
                / task_name
                / "timing_logs"
                / f"{task_name}_ep{episode_id}.json"
            )
            if not timing_path.exists():
                continue
            try:
                payload = json.loads(timing_path.read_text(encoding="utf-8"))
                task_timings[str(episode_id)] = {
                    "status": payload["status"],
                    **payload["summary"],
                }
            except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as e:
                print(f"Warning: cannot include {timing_path} in timing summary: {e}")
        if task_timings:
            episodes[task_name] = task_timings

    return {
        "unit": TIMING_UNIT,
        "definition": TIMING_DEFINITION,
        "episodes": episodes,
    }


def summarize_eval_timing(episode_timings: dict) -> dict:
    """Aggregate every episode call into one seed/repeat evaluation summary."""
    summary = {}
    for component in TIMING_COMPONENTS:
        call_count = 0
        total_ms = 0.0
        for task_timings in episode_timings.values():
            for episode_timing in task_timings.values():
                component_timing = episode_timing.get(component, {})
                call_count += int(component_timing.get("call_count", 0))
                total_ms += float(component_timing.get("total_ms", 0.0))
        summary[component] = {
            "call_count": call_count,
            "average_ms": (
                round(total_ms / call_count, 3) if call_count else None
            ),
            "total_ms": round(total_ms, 3),
        }
    return summary


class EpisodeEvaluator:
    def __init__(self, args: Args, save_dir: Path):
        self.args = args
        self.save_dir = save_dir
        self._active_episode_timing: Optional[dict] = None

    def _start_episode_timing(self, env_runner: EnvRunner) -> None:
        self._active_episode_timing = {
            "task": env_runner.env_id,
            "episode_id": env_runner.episode_id,
            "cycles": [],
        }

    def _record_inference_cycle(
        self,
        *,
        step: int,
        framework_ms: float,
        reporter_ms: Optional[float],
        manager_ms: Optional[float],
        executer_ms: float,
    ) -> None:
        if self._active_episode_timing is None:
            raise RuntimeError("Episode timing was not initialized")
        self._active_episode_timing["cycles"].append(
            {
                "step": int(step),
                "framework_ms": round(framework_ms, 3),
                "reporter_ms": (
                    round(reporter_ms, 3) if reporter_ms is not None else None
                ),
                "manager_ms": (
                    round(manager_ms, 3) if manager_ms is not None else None
                ),
                "executer_ms": round(executer_ms, 3),
            }
        )

    def finish_episode_timing(
        self,
        task_name: str,
        episode_id: int,
        status: str,
    ) -> Optional[Path]:
        """Persist all completed timing cycles, including those before an error."""
        timing = self._active_episode_timing
        if timing is None:
            return None
        if timing["task"] != task_name or timing["episode_id"] != episode_id:
            raise RuntimeError(
                "Attempted to finalize timing for a different episode: "
                f"active={timing['task']}/ep{timing['episode_id']}, "
                f"requested={task_name}/ep{episode_id}"
            )

        payload = {
            "task": task_name,
            "episode_id": episode_id,
            "status": status,
            "unit": TIMING_UNIT,
            "definition": TIMING_DEFINITION,
            "summary": summarize_episode_timing(timing["cycles"]),
            "cycles": timing["cycles"],
        }
        timing_dir = self.save_dir / task_name / "timing_logs"
        timing_dir.mkdir(parents=True, exist_ok=True)
        timing_path = timing_dir / f"{task_name}_ep{episode_id}.json"
        temporary_path = timing_path.with_suffix(".json.tmp")
        with temporary_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        os.replace(temporary_path, timing_path)
        self._active_episode_timing = None
        return timing_path

    def eval_each_episode(
        self,
        env_runner: EnvRunner,
        manager: ManagerBase,
        executer: Executer,
        reporter: ReporterBase,
        video_save_dir: Path,
    ) -> str:
        self._start_episode_timing(env_runner)
        executer.start_episode()
        epstate = EpisodeState()
        task_goal, recorder = self.init_episode(env_runner, epstate, video_save_dir)
        manager.start_episode(epstate, env_runner)
        reporter.start_episode(epstate, env_runner)

        img, wrist_img, robot_state = epstate.get_current_obs()
        observation_received_at_ns = time.perf_counter_ns()
        success_flag = "unknown"
        subgoal = None
        last_subgoal = None
        reporter_result = None

        while True:
            is_inference_cycle = not epstate.action_plan
            framework_started_at_ns = (
                observation_received_at_ns if is_inference_cycle else None
            )
            reporter_ms = None
            manager_ms = None
            should_predict_subgoal = (
                is_inference_cycle
                and (
                    epstate.count % self.args.subgoal_keep_period == 0
                    or last_subgoal is None
                )
            )
            if should_predict_subgoal:
                # Reporter compares the init overall/wrist pair with a rolling
                # window of up to three recent pairs. A successful result
                # clears that window before the result is passed to Manager.
                reporter_will_infer = (
                    self.args.reporter_type != "none" and subgoal is not None
                )
                reporter_started_at_ns = (
                    time.perf_counter_ns() if reporter_will_infer else None
                )
                reporter_result = reporter.step(epstate, subgoal)
                if reporter_started_at_ns is not None:
                    reporter_ms = (
                        time.perf_counter_ns() - reporter_started_at_ns
                    ) / 1_000_000

            # Manager receives/buffers the same current observation only after
            # Reporter has finished the pre-prediction check.
            manager_started_at_ns = (
                time.perf_counter_ns() if should_predict_subgoal else None
            )
            manager.step(epstate)

            if is_inference_cycle:
                if should_predict_subgoal:
                    if manager_started_at_ns is None:
                        raise RuntimeError("Manager timing was not initialized")
                    subgoal, has_api_error = manager.get_subgoal(
                        epstate.count,
                        subgoal,
                        last_subgoal,
                        reporter_result,
                    )
                    manager_ms = (
                        time.perf_counter_ns() - manager_started_at_ns
                    ) / 1_000_000
                else:
                    subgoal = last_subgoal
                    has_api_error = False

                if has_api_error:
                    break

                reporter.observe_subgoal(
                    subgoal,
                    img,
                    wrist_img,
                    epstate.count,
                    reporter_result,
                )
                # Trinity's symbolic Executer receives only the current subgoal.
                # Keep the task goal only for legacy non-symbolic policies, which
                # do not have a Manager or a subgoal input.
                if self.args.subgoal_type in SUBGOAL_TYPES:
                    if subgoal is None:
                        raise RuntimeError(
                            "Symbolic Executer requires a subgoal from Manager"
                        )
                    execution_goal = subgoal
                else:
                    execution_goal = task_goal
                executer_started_at_ns = time.perf_counter_ns()
                action_chunk = executer.get_action_chunk(
                    epstate, img, wrist_img, robot_state, execution_goal,
                    exec_horizon=self.args.obs_horizon
                )
                action_chunk_ready_at_ns = time.perf_counter_ns()
                executer_ms = (
                    action_chunk_ready_at_ns - executer_started_at_ns
                ) / 1_000_000
                if framework_started_at_ns is None:
                    raise RuntimeError("Framework timing was not initialized")
                framework_ms = (
                    action_chunk_ready_at_ns - framework_started_at_ns
                ) / 1_000_000
                self._record_inference_cycle(
                    step=epstate.count,
                    framework_ms=framework_ms,
                    reporter_ms=reporter_ms,
                    manager_ms=manager_ms,
                    executer_ms=executer_ms,
                )

                epstate.action_plan.extend(action_chunk)
                epstate.clear_buffers()

                last_subgoal = subgoal

            action = epstate.action_plan.popleft()
            obs, stop_flag, success_flag = env_runner.step(action)
            observation_received_at_ns = time.perf_counter_ns()
            epstate.count += 1

            if epstate.count > self.args.max_steps:
                success_flag = "timeout"
                break

            img, wrist_img, robot_state = obs

            epstate.add_observation(img, wrist_img, robot_state)
            recorder.record(
                image=img.copy(),
                wrist_image=wrist_img.copy(),
                state=robot_state.copy(),
                action=action.copy(),
                subgoal=subgoal,
            )
            if stop_flag:
                break

        reporter.end_episode(epstate, success_flag)
        if success_flag == "unknown":
            return "unknown"

        video_filename = f"{env_runner.env_id}_ep{env_runner.episode_id}_{success_flag}_{task_goal}_{env_runner.difficulty}.mp4"
        recorder.save_video(video_filename)

        manager.end_episode(epstate, success_flag)
        executer.end_episode()
        return success_flag


    def init_episode(
        self,
        env_runner: EnvRunner,
        epstate: EpisodeState,
        video_save_dir: Path,
    ) -> Tuple[str, RolloutRecorder]:
        pre_traj = env_runner.get_init_obs()
        task_goal = pre_traj["task_goal"]

        recorder = RolloutRecorder(video_save_dir, task_goal, fps=30)

        print(f"task_goal: {task_goal}")

        epstate.image_buffer.extend(pre_traj["images"])
        epstate.wrist_image_buffer.extend(pre_traj["wrist_images"])
        epstate.state_buffer.extend(pre_traj["states"])

        for i in range(len(pre_traj["images"])):
            recorder.record(
                image=pre_traj["images"][i].copy(),
                wrist_image=pre_traj["wrist_images"][i].copy(),
                state=pre_traj["states"][i].copy(),
                is_video_demo=env_runner.env_id in TASK_WITH_VIDEO_DEMO and i < len(pre_traj["images"]) - 1,
                subgoal=None if self.args.subgoal_type is None else "[initializing...]",
            )

        epstate.exec_start_idx = len(epstate.image_buffer) - 1
        print(f"exec_start_idx: {epstate.exec_start_idx}")
        return task_goal, recorder

def get_evaluation_run_name(args: Args) -> str:
    if args.run_name:
        return args.run_name
    if args.subgoal_type in SUBGOAL_TYPES:
        if args.manager_use_gemini:
            return "gemini"
        if args.manager_use_qwenvl:
            return "qwenvl"
        if args.manager_use_memer:
            return "memer"
        return "oracle"
    return "default"


def get_evaluation_run_directory(args: Args) -> Path:
    """Return the directory shared by every seed and repeat in one run."""
    return Path(args.save_dir) / args.executer_name / get_evaluation_run_name(args)


def format_eval_duration(seconds: float) -> str:
    """Format an evaluation duration without zero-valued leading units."""
    total_seconds = max(0, round(seconds))
    hours, remaining_seconds = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remaining_seconds, 60)
    if hours:
        return f"{hours}h {minutes}min {seconds}s"
    if minutes:
        return f"{minutes}min {seconds}s"
    return f"{seconds}s"


def parse_eval_duration(value: object) -> float:
    """Parse the current display format or a legacy numeric-seconds value."""
    if isinstance(value, (int, float)):
        duration_seconds = float(value)
    elif isinstance(value, str):
        duration_seconds = 0.0
        for part in value.split():
            if part.endswith("min"):
                duration_seconds += float(part[:-3]) * 60
            elif part.endswith("h"):
                duration_seconds += float(part[:-1]) * 3600
            elif part.endswith("s"):
                duration_seconds += float(part[:-1])
            else:
                raise ValueError(f"unknown duration unit in {part!r}")
    else:
        raise TypeError(f"unsupported duration value: {value!r}")

    if duration_seconds < 0:
        raise ValueError("duration cannot be negative")
    return duration_seconds


def setup_save_directory(args: Args) -> Path:
    """Set up one seed/repeat directory and validate overwrite behavior."""
    if args.repeat_id < 1:
        raise ValueError(f"repeat_id must be at least 1, got {args.repeat_id}")

    save_dir = (
        get_evaluation_run_directory(args)
        / f"seed{args.executer_seed}"
        / f"repeat{args.repeat_id}"
    )

    if save_dir.exists():
        if args.overwrite:
            shutil.rmtree(save_dir)
            print(f"we will overwrite the evaluation at {save_dir}")
        else:
            print("we will resume the evaluation")

    save_dir.mkdir(parents=True, exist_ok=True)
    return save_dir


def setup_prompt_metadata(args: Args, save_dir: Path) -> dict:
    """Persist resolved prompt identities and prevent mixed-version resumes."""
    metadata = {
        "manager": (
            get_manager_prompt(args.manager_prompt_version).metadata()
            if args.manager_use_qwenvl
            else None
        ),
        "reporter": (
            get_reporter_prompt(args.reporter_prompt_version).metadata()
            if args.reporter_type == "qwenvl"
            else None
        ),
    }
    metadata_path = save_dir / "prompt_metadata.json"
    if metadata_path.exists():
        existing = json.loads(metadata_path.read_text(encoding="utf-8"))
        if existing != metadata:
            raise ValueError(
                "Cannot resume an evaluation directory with different prompt "
                f"versions: existing={existing}, requested={metadata}. Choose "
                "a new run name or enable overwrite."
            )
    else:
        legacy_artifacts = [
            path.name
            for path in (save_dir / "progress.json", save_dir / "log.json")
            if path.exists()
        ]
        if legacy_artifacts:
            raise ValueError(
                "Cannot safely resume an evaluation created before prompt "
                f"metadata was recorded ({legacy_artifacts}). Choose a new run "
                "name or enable overwrite."
            )
        metadata_path.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    return metadata


def _parse_nvidia_smi_int(value: str) -> Optional[int]:
    value = value.strip()
    if value.lower() in {"", "n/a", "[not supported]"}:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _gpu_selectors(value: str) -> list[str]:
    value = value.strip()
    if not value or value.lower() in {"none", "nodevfiles", "-1"}:
        return []
    if value.lower() == "all":
        return ["all"]
    return [selector.strip() for selector in value.split(",") if selector.strip()]


GPU_QUERY_FIELDS = (
    "index",
    "uuid",
    "memory.total",
    "memory.used",
)
GPU_PROCESS_QUERY_FIELDS = ("gpu_uuid", "pid", "used_gpu_memory")
GPU_SAMPLE_INTERVAL_SECONDS = 5.0


def _gpu_assignments(args: Args) -> dict[str, list[str]]:
    return {
        "executer": _gpu_selectors(args.executer_gpu_id),
        "manager_reporter": _gpu_selectors(args.manager_reporter_gpu_id),
    }


def _parse_gpu_devices(
    raw_csv: str,
    assignments: dict[str, list[str]],
) -> list[dict]:
    devices = []
    for row in csv.reader(raw_csv.splitlines(), skipinitialspace=True):
        if len(row) != len(GPU_QUERY_FIELDS):
            continue
        index, uuid, memory_total, memory_used = (value.strip() for value in row)
        parsed_index = _parse_nvidia_smi_int(index)
        if parsed_index is None:
            continue
        selected = any(
            "all" in selectors
            or any(
                selector == index
                or selector == uuid
                or uuid.startswith(selector)
                for selector in selectors
            )
            for selectors in assignments.values()
        )
        if not selected:
            continue
        devices.append(
            {
                "gpu_id": parsed_index,
                "uuid": uuid,
                "total_memory_mib": _parse_nvidia_smi_int(memory_total),
                "start_used_mib": _parse_nvidia_smi_int(memory_used),
            }
        )
    return devices


def _process_parent_map() -> dict[int, int]:
    try:
        completed = subprocess.run(
            ["ps", "-eo", "pid=,ppid="],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return {}
    parents = {}
    for line in completed.stdout.splitlines():
        columns = line.split()
        if len(columns) != 2:
            continue
        try:
            pid, parent_pid = (int(value) for value in columns)
        except ValueError:
            continue
        parents[pid] = parent_pid
    return parents


def _belongs_to_eval_process(
    pid: int,
    root_pids: set[int],
    parent_by_pid: dict[int, int],
) -> bool:
    visited = set()
    while pid > 1 and pid not in visited:
        if pid in root_pids:
            return True
        visited.add(pid)
        pid = parent_by_pid.get(pid, 0)
    return False


def _query_eval_gpu_memory(devices: list[dict]) -> tuple[dict[str, int], Optional[str]]:
    if any(device["uuid"].startswith("index:") for device in devices):
        return {}, "GPU UUID was unavailable"
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        return {}, "nvidia-smi was not found"
    try:
        completed = subprocess.run(
            [
                nvidia_smi,
                f"--query-compute-apps={','.join(GPU_PROCESS_QUERY_FIELDS)}",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as e:
        return {}, str(e)

    root_pids = {os.getpid()}
    executer_pid = _parse_nvidia_smi_int(
        os.environ.get("ROBOMME_EXECUTER_LAUNCH_PID", "")
    )
    if executer_pid is not None:
        root_pids.add(executer_pid)
    parent_by_pid = _process_parent_map()
    memory_by_uuid = {device["uuid"]: 0 for device in devices}
    for row in csv.reader(completed.stdout.splitlines(), skipinitialspace=True):
        if len(row) != len(GPU_PROCESS_QUERY_FIELDS):
            continue
        uuid, pid, memory_used = (value.strip() for value in row)
        parsed_pid = _parse_nvidia_smi_int(pid)
        parsed_memory = _parse_nvidia_smi_int(memory_used)
        if (
            uuid in memory_by_uuid
            and parsed_pid is not None
            and parsed_memory is not None
            and _belongs_to_eval_process(parsed_pid, root_pids, parent_by_pid)
        ):
            memory_by_uuid[uuid] += parsed_memory
    return memory_by_uuid, None


def _fallback_gpu_devices(assignments: dict[str, list[str]]) -> list[dict]:
    gpu_ids = sorted(
        {
            int(selector)
            for selectors in assignments.values()
            for selector in selectors
            if selector.isdigit()
        }
    )
    return [
        {
            "gpu_id": gpu_id,
            "uuid": f"index:{gpu_id}",
            "total_memory_mib": None,
            "start_used_mib": None,
        }
        for gpu_id in gpu_ids
    ]


def _format_gpu_memory(memory_mib: Optional[float]) -> Optional[str]:
    return f"{memory_mib / 1024:.1f} GB" if memory_mib is not None else None


class GpuUsageMonitor:
    """Measure only this seed/repeat evaluation's GPU memory."""

    def __init__(self, args: Args, interval_seconds: float = GPU_SAMPLE_INTERVAL_SECONDS):
        assignments = _gpu_assignments(args)
        self.devices = _parse_gpu_devices(
            os.environ.get("ROBOMME_GPU_START_SNAPSHOT", ""),
            assignments,
        )
        if not self.devices:
            self.devices = _fallback_gpu_devices(assignments)
        self.interval_seconds = interval_seconds
        self._samples_by_uuid: dict[str, list[int]] = {
            device["uuid"]: [] for device in self.devices
        }
        self._stop_event = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name="robomme-gpu-monitor",
            daemon=True,
        )

    def _sample(self) -> None:
        memory_by_uuid, error = _query_eval_gpu_memory(self.devices)
        if error is not None:
            return
        for uuid, memory_used in memory_by_uuid.items():
            self._samples_by_uuid[uuid].append(memory_used)

    def _run(self) -> None:
        while not self._stop_event.wait(self.interval_seconds):
            self._sample()

    def start(self) -> None:
        self._sample()
        self._thread.start()

    def stop(self) -> list[dict]:
        self._stop_event.set()
        self._thread.join(timeout=self.interval_seconds + 10)
        self._sample()
        result = []
        for device in self.devices:
            samples = self._samples_by_uuid[device["uuid"]]
            average_used_mib = sum(samples) / len(samples) if samples else None
            start_used_mib = device["start_used_mib"]
            result.append(
                {
                    "gpu_id": device["gpu_id"],
                    "total_memory": _format_gpu_memory(
                        device["total_memory_mib"]
                    ),
                    "start_used": _format_gpu_memory(start_used_mib),
                    "average_eval_used": _format_gpu_memory(average_used_mib),
                }
            )
        return result


def update_run_summary(args: Args) -> Path:
    """Aggregate all completed seed/repeat logs for the current run."""
    run_dir = get_evaluation_run_directory(args)
    seed_values = [
        int(value.strip())
        for value in (args.eval_seeds or str(args.executer_seed)).split(",")
        if value.strip()
    ]
    if not seed_values:
        raise ValueError("eval_seeds must contain at least one seed")
    if args.num_repeats < 1:
        raise ValueError(f"num_repeats must be at least 1, got {args.num_repeats}")

    completed = []
    missing = []
    task_success_rates: dict[str, list[float]] = {}
    eval_durations: list[float] = []
    timing_call_counts = {component: 0 for component in TIMING_COMPONENTS}
    timing_totals_ms = {component: 0.0 for component in TIMING_COMPONENTS}
    for seed in seed_values:
        for repeat_id in range(1, args.num_repeats + 1):
            log_path = run_dir / f"seed{seed}" / f"repeat{repeat_id}" / "log.json"
            if not log_path.exists():
                missing.append({"seed": seed, "repeat": repeat_id})
                continue
            try:
                with log_path.open("r", encoding="utf-8") as f:
                    result = json.load(f)
                total_success_rate = float(result["total_success_rate"])
                per_task = result["success_rate"]
            except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as e:
                print(f"Warning: cannot include {log_path} in summary: {e}")
                missing.append({"seed": seed, "repeat": repeat_id})
                continue

            raw_eval_duration = result.get(
                "eval_duration",
                result.get("eval_duration_seconds"),
            )
            eval_duration_seconds = None
            if raw_eval_duration is not None:
                try:
                    eval_duration_seconds = parse_eval_duration(raw_eval_duration)
                except (TypeError, ValueError) as e:
                    print(
                        f"Warning: ignoring invalid eval duration in {log_path}: {e}"
                    )
                    eval_duration_seconds = None

            completed_run = {
                "seed": seed,
                "repeat": repeat_id,
                "total_success_rate": total_success_rate,
                "eval_duration": (
                    format_eval_duration(eval_duration_seconds)
                    if eval_duration_seconds is not None
                    else None
                ),
            }
            if "gpu" in result:
                completed_run["gpu"] = result["gpu"]
            run_episode_timing = result.get("episode_timing", {}).get(
                "episodes",
                {},
            )
            if run_episode_timing:
                run_timing = summarize_eval_timing(run_episode_timing)
                completed_run["average_timing_ms"] = {
                    component: run_timing[component]["average_ms"]
                    for component in TIMING_COMPONENTS
                }
                for component in TIMING_COMPONENTS:
                    timing_call_counts[component] += run_timing[component][
                        "call_count"
                    ]
                    timing_totals_ms[component] += run_timing[component][
                        "total_ms"
                    ]
            completed.append(completed_run)
            if eval_duration_seconds is not None:
                eval_durations.append(eval_duration_seconds)
            for task_name, success_rate in per_task.items():
                task_success_rates.setdefault(task_name, []).append(float(success_rate))

    summary = {
        "executer_name": args.executer_name,
        "executer_ckpt_id": args.executer_ckpt_id,
        "run_name": get_evaluation_run_name(args),
        "manager_prompt_version": (
            get_manager_prompt(args.manager_prompt_version).version
            if args.manager_use_qwenvl
            else None
        ),
        "manager_prompt_sha256": (
            get_manager_prompt(args.manager_prompt_version).content_hash
            if args.manager_use_qwenvl
            else None
        ),
        "reporter_prompt_version": (
            get_reporter_prompt(args.reporter_prompt_version).version
            if args.reporter_type == "qwenvl"
            else None
        ),
        "reporter_prompt_sha256": (
            get_reporter_prompt(args.reporter_prompt_version).content_hash
            if args.reporter_type == "qwenvl"
            else None
        ),
        "reporter_history_size": args.reporter_history_size,
        "reporter_debounce": args.reporter_debounce,
        "reporter_gripper_gate": {
            "enabled": args.reporter_gripper_gate,
            "closed_position": args.reporter_gripper_closed_position,
            "open_position": args.reporter_gripper_open_position,
            "endpoint_buffer": args.reporter_gripper_endpoint_buffer,
        },
        "seeds": seed_values,
        "repeats_per_seed": args.num_repeats,
        "expected_run_count": len(seed_values) * args.num_repeats,
        "completed_run_count": len(completed),
        "missing_runs": missing,
        "total_average_success_rate": (
            sum(item["total_success_rate"] for item in completed) / len(completed)
            if completed
            else None
        ),
        "average_eval_duration": (
            format_eval_duration(sum(eval_durations) / len(eval_durations))
            if eval_durations
            else None
        ),
        "average_timing_ms": {
            component: (
                round(
                    timing_totals_ms[component]
                    / timing_call_counts[component],
                    3,
                )
                if timing_call_counts[component]
                else None
            )
            for component in TIMING_COMPONENTS
        },
        "per_task_average_success_rate": {
            task_name: sum(values) / len(values)
            for task_name, values in sorted(task_success_rates.items())
        },
        "runs": completed,
    }

    summary_path = run_dir / "summary.json"
    temporary_path = run_dir / "summary.json.tmp"
    with temporary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    os.replace(temporary_path, summary_path)
    print(
        f"[robomme] Updated {summary_path}: "
        f"{len(completed)}/{summary['expected_run_count']} runs completed, "
        f"total average success rate={summary['total_average_success_rate']}, "
        f"average eval duration="
        f"{summary['average_eval_duration']}"
    )
    return summary_path


def setup_log_dict(save_dir: Path, args: Args) -> dict:
    if os.path.exists(save_dir / "progress.json"):
        with open(save_dir / "progress.json", "r") as f:
            log_dict = json.load(f)

    elif os.path.exists(save_dir / "log.json"):
        with open(save_dir / "log.json", "r") as f:
            log_dict = json.load(f)
        log_dict.pop("success_rate", None)
        log_dict.pop("total_success_rate", None)
    else:
        log_dict = {}

    for task_name, episode_results in log_dict.items():
        # JSON object keys are strings after a resume, while freshly recorded
        # episode IDs used to be integers. Keep one representation so the
        # completed-episode check behaves the same before and after a restart.
        log_dict[task_name] = {
            str(episode_id): result
            for episode_id, result in episode_results.items()
            if result != "error"
        }

    if args.re_eval_tasks:
        for task_name in args.re_eval_tasks.split(","):
            if task_name in log_dict:
                del log_dict[task_name]
                task_dir = save_dir / task_name
                if task_dir.exists():
                    shutil.rmtree(task_dir)

    with open(save_dir / "progress.json", "w") as f:
        json.dump(log_dict, f, indent=2)

    return log_dict


def evaluate(args: Args):
    """Main evaluation function."""
    eval_started_at = time.perf_counter()
    args.manager_qwen_backbone_path = resolve_qwen_backbone_path(
        args.manager_qwen_backbone_name,
        model_root=args.manager_qwen_backbone_root,
        model_path=args.manager_qwen_backbone_path or None,
    )
    args.reporter_qwen_backbone_path = resolve_qwen_backbone_path(
        args.reporter_qwen_backbone_name,
        model_root=args.reporter_qwen_backbone_root,
        model_path=args.reporter_qwen_backbone_path or None,
    )
    print(
        f"[robomme] Manager Qwen backbone: {args.manager_qwen_backbone_name} "
        f"({args.manager_qwen_backbone_path})"
    )
    print(
        f"[robomme] Reporter Qwen backbone: {args.reporter_qwen_backbone_name} "
        f"({args.reporter_qwen_backbone_path})"
    )
    check_args(args)

    save_dir = setup_save_directory(args)
    prompt_metadata = setup_prompt_metadata(args, save_dir)

    log_dict = setup_log_dict(save_dir, args)

    if args.only_tasks:
        task_names = args.only_tasks.split(",")
    else:
        task_names = TASK_NAME_LIST

    if args.exclude_tasks:
        task_names = [task_name for task_name in task_names if task_name not in args.exclude_tasks.split(",")]
        for task in args.exclude_tasks.split(","):
            log_dict[task] = {str(i): False for i in range(50)}

    requested_episode_ids = (
        [int(value.strip()) for value in args.episode_ids.split(",")]
        if args.episode_ids
        else None
    )
    episodes_per_task = (
        len(requested_episode_ids)
        if requested_episode_ids is not None
        else args.num_episodes
    )
    episodes_per_run = len(task_names) * episodes_per_task
    seed_values = [
        int(value.strip())
        for value in (args.eval_seeds or str(args.executer_seed)).split(",")
        if value.strip()
    ]
    if args.executer_seed not in seed_values:
        raise ValueError(
            f"Current seed {args.executer_seed} is not present in eval_seeds "
            f"{seed_values}"
        )
    run_index = (
        seed_values.index(args.executer_seed) * args.num_repeats
        + args.repeat_id
        - 1
    )
    total_episodes = episodes_per_run * len(seed_values) * args.num_repeats
    run_episode_offset = run_index * episodes_per_run
    selected_episode_ids = (
        requested_episode_ids
        if requested_episode_ids is not None
        else list(range(args.num_episodes))
    )
    completed_current_run = sum(
        1
        for task_name in task_names
        for episode_id in selected_episode_ids
        if str(episode_id) in log_dict.get(task_name, {})
        or episode_id in log_dict.get(task_name, {})
    )
    final_log_path = save_dir / "log.json"
    if completed_current_run == episodes_per_run and final_log_path.exists():
        print(
            f"[robomme] Evaluation already complete at {final_log_path}; "
            "leaving the existing result and duration unchanged."
        )
        update_run_summary(args)
        return

    gpu_monitor = GpuUsageMonitor(args)
    gpu_monitor.start()
    manager = build_manager(args, save_dir)
    executer = build_executer(args)
    reporter = build_reporter(args, save_dir)
    evaluator = EpisodeEvaluator(args, save_dir)
    evaluation_progress = tqdm(
        total=total_episodes,
        initial=run_episode_offset + completed_current_run,
        desc=f"seed={args.executer_seed} repeat={args.repeat_id}/{args.num_repeats}",
        unit="episode",
        dynamic_ncols=True,
        leave=True,
        file=sys.stderr,
        bar_format=(
            "{desc} |{bar}| {n_fmt}/{total_fmt} episodes "
            "({percentage:3.0f}%) [elapsed={elapsed}, remaining={remaining}, {rate_fmt}]"
        ),
    )

    # log.json summarizes the latest completed run. Remove only this derived
    # summary so a new invocation can add tasks/episodes from progress.json.
    # Existing task videos, frames, logs, and episode results remain untouched.
    if final_log_path.exists():
        final_log_path.unlink()

    while not os.path.exists(save_dir / "log.json"):
        for task_name in task_names:
            if task_name not in log_dict:
                log_dict[task_name] = {}

            video_save_dir = save_dir / task_name / "videos"
            env_runner = EnvRunner(task_name, video_save_dir, max_steps=args.max_steps)
            if requested_episode_ids is not None:
                episode_ids = requested_episode_ids
                invalid_ids = [
                    episode_id
                    for episode_id in episode_ids
                    if episode_id < 0 or episode_id >= env_runner.num_episodes
                ]
                if invalid_ids:
                    raise ValueError(
                        f"Invalid episode IDs {invalid_ids} for task {task_name}; "
                        f"valid range is 0-{env_runner.num_episodes - 1}"
                    )
            else:
                episode_ids = range(args.num_episodes)

            success_flag = "unknown"

            for episode_id in episode_ids:
                episode_key = str(episode_id)
                if episode_key in log_dict[task_name]:
                    continue

                success_flag = "unknown"
                episode_error = None
                with manager_log(
                    save_dir, task_name, episode_id, args.save_manager_logs
                ):
                    try:
                        env_runner.make_env(episode_id)
                        print(f"\n[robomme] env for task {task_name} episode {episode_id} setup finished")
                        success_flag = evaluator.eval_each_episode(
                            env_runner, manager, executer, reporter, video_save_dir
                        )
                        if success_flag == "unknown":
                            log_dict[task_name][episode_key] = "error"
                        else:
                            log_dict[task_name][episode_key] = success_flag == "success"
                    except Exception as e:
                        print(f"Error evaluating episode {episode_id} for task {task_name}: {e}")
                        traceback.print_exc()
                        log_dict[task_name][episode_key] = "error"
                        episode_error = e
                    finally:
                        try:
                            evaluator.finish_episode_timing(
                                task_name,
                                episode_id,
                                "error" if episode_error is not None else success_flag,
                            )
                        finally:
                            env_runner.close_env()

                if episode_error is not None:
                    print(
                        f"[robomme] Episode failed: task={task_name}, "
                        f"episode={episode_id}: {episode_error}. See the Manager "
                        "log for the traceback."
                    )

                evaluation_progress.update(1)

                with open(save_dir / "progress.json", "w") as f:
                    json.dump(log_dict, f, indent=2)

            del env_runner
            time.sleep(1)

        invalid_episode_results = [
            (task_name, episode_id, result)
            for task_name, episode_results in log_dict.items()
            for episode_id, result in episode_results.items()
            if not isinstance(result, bool)
        ]
        if invalid_episode_results:
            evaluation_progress.close()
            invalid_text = ", ".join(
                f"{task_name}/{episode_id}={result!r}"
                for task_name, episode_id, result in invalid_episode_results
            )
            raise RuntimeError(
                f"Evaluation finished with {len(invalid_episode_results)} "
                f"invalid episode result(s): {invalid_text}. Re-run with "
                "overwrite disabled to evaluate only entries marked 'error'."
            )

        try:
            final_results = {}
            final_results["success_rate"] = {
                task_name: sum(log_dict[task_name].values()) / len(log_dict[task_name].values())
                for task_name in log_dict.keys()
            }
            final_results["total_success_rate"] = (
                sum(final_results["success_rate"].values()) / len(final_results["success_rate"].values())
            )
            final_results["executer_seed"] = args.executer_seed
            final_results["repeat_id"] = args.repeat_id
            final_results["executer_ckpt_id"] = args.executer_ckpt_id
            final_results["eval_duration"] = format_eval_duration(
                time.perf_counter() - eval_started_at
            )
            final_results["qwen_backbones"] = {
                "manager": {
                    "name": args.manager_qwen_backbone_name,
                    "path": args.manager_qwen_backbone_path,
                },
                "reporter": {
                    "name": args.reporter_qwen_backbone_name,
                    "path": args.reporter_qwen_backbone_path,
                },
            }
            final_results["prompts"] = prompt_metadata
            final_results["reporter_runtime"] = {
                "history_size": args.reporter_history_size,
                "debounce": args.reporter_debounce,
                "gripper_gate": {
                    "enabled": args.reporter_gripper_gate,
                    "closed_position": args.reporter_gripper_closed_position,
                    "open_position": args.reporter_gripper_open_position,
                    "endpoint_buffer": args.reporter_gripper_endpoint_buffer,
                },
            }
            final_results["gpu"] = gpu_monitor.stop()
            final_results["episode_timing"] = load_episode_timing_summaries(
                save_dir,
                log_dict,
            )
            with open(save_dir / "log.json", "w") as f:
                json.dump(final_results, f, indent=2)
            update_run_summary(args)
        except Exception as e:
            print(f"Error saving final results: {e}")
            evaluation_progress.close()
            raise

    evaluation_progress.close()


if __name__ == "__main__":
    import tyro
    tyro.cli(evaluate)
