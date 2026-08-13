import dataclasses
import json
import os
import shutil
import sys
import time
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Optional, Any, TextIO, Tuple

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



@dataclasses.dataclass
class Args:
    executer_host: str = "0.0.0.0"
    executer_port: int = 8011

    obs_horizon: int = 16
    max_steps: int = 1300
    save_dir: str = "runs/evaluation"
    run_name: str = ""
    overwrite: bool = True
    save_episode_logs: bool = True

    executer_use_history: bool = True
    executer_name: str = "symbolic-grounded-subgoal"
    executer_seed: int = 7
    executer_ckpt_id: int = 79999

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
    manager_qwenvl_simpleSG_adapter_path: str = "runs/ckpts/vlm_subgoal_predictor/qwenvl/simple_subgoal/checkpoint-1400"
    manager_qwenvl_groundSG_adapter_path: str = "runs/ckpts/vlm_subgoal_predictor/qwenvl/grounded_subgoal/checkpoint-1200"
    manager_memer_adapter_path: str = "runs/ckpts/vlm_subgoal_predictor/memer/grounded_subgoal/checkpoint-1300"
    manager_save_memer_kf: bool = False
    subgoal_keep_period: int = 1 # ever subgoal should be kept for this many steps

    # Reporter
    reporter_type: str = "qwenvl"
    reporter_model_path: str = "Qwen/Qwen3-VL-4B-Instruct"
    # this can accelerate the evaluation process for symbolic memory
    # In our experiments, we just set this to 1
    num_episodes: int = 10 # number of episodes to evaluate for each task
    episode_ids: str = "2" # exact episode IDs to evaluate, e.g. "7" or "2,7"; overrides num_episodes


class TeeStream:
    """Write output to both the original terminal stream and a log file."""

    def __init__(self, terminal: TextIO, log_file: TextIO):
        self.terminal = terminal
        self.log_file = log_file

    def write(self, text: str) -> int:
        self.terminal.write(text)
        self.log_file.write(text)
        return len(text)

    def flush(self) -> None:
        self.terminal.flush()
        self.log_file.flush()


@contextmanager
def episode_log(save_dir: Path, task_name: str, episode_id: int, enabled: bool):
    """Capture one episode's stdout/stderr while preserving terminal output."""
    if not enabled:
        yield None
        return

    log_dir = save_dir / task_name / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{task_name}_ep{episode_id}.log"
    with log_path.open("w", encoding="utf-8", buffering=1) as log_file:
        stdout_tee = TeeStream(sys.stdout, log_file)
        stderr_tee = TeeStream(sys.stderr, log_file)
        with redirect_stdout(stdout_tee), redirect_stderr(stderr_tee):
            print(f"[robomme] episode log: {log_path}")
            yield log_path



class EpisodeEvaluator:
    def __init__(self, args: Args, save_dir: Path):
        self.args = args
        self.save_dir = save_dir

    def eval_each_episode(
        self,
        env_runner: EnvRunner,
        manager: ManagerBase,
        executer: Executer,
        reporter: ReporterBase,
        video_save_dir: Path,
    ) -> str:
        executer.start_episode()
        epstate = EpisodeState()
        task_goal, recorder = self.init_episode(env_runner, epstate, video_save_dir)
        manager.start_episode(epstate, env_runner)
        reporter.start_episode(epstate, env_runner)

        img, wrist_img, robot_state = epstate.get_current_obs()
        prompt = task_goal
        success_flag = "unknown"
        subgoal = None
        last_subgoal = None
        reporter_result = None

        while True:
            manager.step(epstate)

            if not epstate.action_plan:
                if epstate.count % self.args.subgoal_keep_period == 0 or last_subgoal is None:
                    subgoal, has_api_error = manager.get_subgoal(
                        epstate.count,
                        subgoal,
                        last_subgoal,
                        reporter_result,
                    )
                else:
                    subgoal = last_subgoal
                    has_api_error = False

                if has_api_error:
                    break

                reporter.observe_subgoal(subgoal, img)
                action_chunk = executer.get_action_chunk(
                    epstate, img, wrist_img, robot_state, prompt, subgoal,
                    exec_horizon=self.args.obs_horizon
                )

                epstate.action_plan.extend(action_chunk)
                epstate.clear_buffers()

                last_subgoal = subgoal

            action = epstate.action_plan.popleft()
            obs, stop_flag, success_flag = env_runner.step(action)
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
            # Ask Reporter once after each Executer action chunk. The result is
            # included in the next QwenVL Manager request.
            if not epstate.action_plan:
                reporter_result = reporter.step(epstate, subgoal)

            if stop_flag:
                break

        if success_flag == "unknown":
            return "unknown"

        video_filename = f"{env_runner.env_id}_ep{env_runner.episode_id}_{success_flag}_{task_goal}_{env_runner.difficulty}.mp4"
        recorder.save_video(video_filename)

        manager.end_episode(epstate, success_flag)
        reporter.end_episode(epstate, success_flag)
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

def setup_save_directory(args: Args) -> Path:
    """Set up and validate save directories."""
    save_dir = (
        Path(args.save_dir)
        / args.executer_name
        / f"ckpt{args.executer_ckpt_id}"
        / f"seed{args.executer_seed}"
    )

    if args.run_name:
        save_dir = save_dir / args.run_name
    elif args.subgoal_type in SUBGOAL_TYPES:
        if args.manager_use_gemini:
            save_dir = save_dir / "gemini"
        elif args.manager_use_qwenvl:
            save_dir = save_dir / "qwenvl"
        elif args.manager_use_memer:
            save_dir = save_dir / "memer"
        else:
            save_dir = save_dir / "oracle"

    if save_dir.exists():
        if args.overwrite:
            shutil.rmtree(save_dir)
            print(f"we will overwrite the evaluation at {save_dir}")
        else:
            print("we will resume the evaluation")

    save_dir.mkdir(parents=True, exist_ok=True)
    return save_dir


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

    for task_name in log_dict:
        error_list = []
        for k, v in log_dict[task_name].items():
            if v == "error":
                error_list.append(k)
        for k in error_list:
            log_dict[task_name].pop(k)

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
    check_args(args)

    save_dir = setup_save_directory(args)

    log_dict = setup_log_dict(save_dir, args)

    if args.only_tasks:
        task_names = args.only_tasks.split(",")
    else:
        task_names = TASK_NAME_LIST

    if args.exclude_tasks:
        task_names = [task_name for task_name in task_names if task_name not in args.exclude_tasks.split(",")]
        for task in args.exclude_tasks.split(","):
            log_dict[task] = {str(i): False for i in range(50)}

    manager = build_manager(args, save_dir)
    executer = build_executer(args)
    reporter = build_reporter(args, save_dir)
    evaluator = EpisodeEvaluator(args, save_dir)

    # log.json summarizes the latest completed run. Remove only this derived
    # summary so a new invocation can add tasks/episodes from progress.json.
    # Existing task videos, frames, logs, and episode results remain untouched.
    final_log_path = save_dir / "log.json"
    if final_log_path.exists():
        final_log_path.unlink()

    while not os.path.exists(save_dir / "log.json"):
        for task_name in task_names:
            if task_name not in log_dict:
                log_dict[task_name] = {}

            video_save_dir = save_dir / task_name / "videos"
            env_runner = EnvRunner(task_name, video_save_dir, max_steps=args.max_steps)
            if args.episode_ids:
                episode_ids = [int(value.strip()) for value in args.episode_ids.split(",")]
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
                if str(episode_id) in log_dict[task_name]:
                    print(f"[robomme] episode {episode_id} already evaluated, skipping...")
                    continue

                with episode_log(
                    save_dir, task_name, episode_id, args.save_episode_logs
                ):
                    try:
                        env_runner.make_env(episode_id)
                        print(f"\n[robomme] env for task {task_name} episode {episode_id} setup finished")
                        success_flag = evaluator.eval_each_episode(
                            env_runner, manager, executer, reporter, video_save_dir
                        )
                        if success_flag == "unknown":
                            log_dict[task_name][episode_id] = "error"
                        else:
                            log_dict[task_name][episode_id] = success_flag == "success"
                    except Exception as e:
                        print(f"Error evaluating episode {episode_id} for task {task_name}: {e}")
                        log_dict[task_name][episode_id] = "error"
                    finally:
                        env_runner.close_env()

                with open(save_dir / "progress.json", "w") as f:
                    json.dump(log_dict, f, indent=2)

                if success_flag == "unknown":
                    print("API calling error, aborting...")
                    return

            del env_runner
            time.sleep(1)

        try:
            final_results = {}
            final_results["success_rate"] = {
                task_name: sum(log_dict[task_name].values()) / len(log_dict[task_name].values())
                for task_name in log_dict.keys()
            }
            final_results["total_success_rate"] = (
                sum(final_results["success_rate"].values()) / len(final_results["success_rate"].values())
            )
            with open(save_dir / "log.json", "w") as f:
                json.dump(final_results, f, indent=2)
        except Exception as e:
            print(f"Error saving final results: {e}")
            time.sleep(1)


if __name__ == "__main__":
    import tyro
    tyro.cli(evaluate)
