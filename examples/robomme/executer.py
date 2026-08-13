"""Executer interface for action-policy inference during RoboMME evaluation."""

import time
from typing import Optional

import numpy as np

from openpi_client import websocket_client_policy as _websocket_client_policy

from utils import EpisodeState, pack_buffer


class Executer:
    """Run the configured action policy through its websocket server.

    The policy implementation and checkpoint are intentionally selected by the
    server configuration in ``scripts/eval.sh``. This client does not assume a
    particular pi model version.
    """

    def __init__(self, args):
        self.args = args
        self.client: Optional[
            _websocket_client_policy.MMEVLAWebsocketClientPolicy
        ] = None

    def start_episode(self) -> None:
        self.client = _websocket_client_policy.MMEVLAWebsocketClientPolicy(
            self.args.executer_host,
            self.args.executer_port,
        )
        resp = self.client.reset()
        while not resp.get("reset_finished", False):
            time.sleep(0.1)

    def get_action_chunk(
        self,
        state: EpisodeState,
        img: np.ndarray,
        wrist_img: np.ndarray,
        robot_state: np.ndarray,
        execution_goal: str,
        exec_horizon: int,
    ) -> np.ndarray:
        if self.client is None:
            raise RuntimeError("Executer.start_episode() must be called before inference")

        if self.args.executer_use_history:
            resp = self.client.add_buffer(
                pack_buffer(
                    state.image_buffer,
                    state.state_buffer,
                    state.exec_start_idx,
                )
            )
            while not resp.get("add_buffer_finished", False):
                time.sleep(0.1)

        element = {
            "observation/image": img,
            "observation/wrist_image": wrist_img,
            "observation/state": robot_state,
            "prompt": execution_goal,
        }

        if self.args.subgoal_type in ("simple_subgoal", "grounded_subgoal"):
            element["simple_subgoal"] = execution_goal
            element["grounded_subgoal"] = execution_goal

        action_chunk = self.client.infer(element)["actions"]
        return action_chunk[:exec_horizon]

    def end_episode(self) -> None:
        self.client = None


def build_executer(args) -> Executer:
    return Executer(args)
