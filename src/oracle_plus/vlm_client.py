"""WebSocket client for the Qwen3.8 Oracle Plus inference service."""

from __future__ import annotations

import socket
import time
from typing import Any

from openpi_client.websocket_client_policy import WebsocketClientPolicy
from websockets.exceptions import ConnectionClosed

from .vlm import SubgoalDecision


class VLMSubgoalPlannerClient:
    """Expose the remote 27B service through the local planner interface."""

    def __init__(self, host: str, port: int) -> None:
        self.host = host
        self.port = port
        self._client = self._connect()
        self.server_metadata = self._client.get_server_metadata()

    def _connect(self) -> WebsocketClientPolicy:
        return WebsocketClientPolicy(host=self.host, port=self.port)

    def _request(
        self,
        payload: dict[str, Any],
        *,
        reconnect_once: bool = True,
    ) -> dict[str, Any]:
        try:
            response = self._client.infer(payload)
        except ConnectionClosed:
            if not reconnect_once:
                raise
            try:
                with socket.create_connection((self.host, self.port), timeout=2):
                    pass
            except OSError as error:
                raise ConnectionError(
                    f"VLM server is no longer listening at {self.host}:{self.port}"
                ) from error
            self._client = self._connect()
            self.server_metadata = self._client.get_server_metadata()
            response = self._client.infer(payload)
        if not isinstance(response, dict):
            raise ValueError("VLM server response is not an object")
        return response

    def infer(
        self,
        *,
        messages: list[dict[str, str]],
        images: list[str],
        include_description: bool,
    ) -> tuple[SubgoalDecision, str, float, float]:
        response = self._request(
            {
                "operation": "infer",
                "messages": messages,
                "images": images,
                "include_description": include_description,
            }
        )
        client_parse_started = time.perf_counter()
        decision_value = response.get("decision")
        if not isinstance(decision_value, dict):
            raise ValueError("VLM server response has no decision object")
        decision = SubgoalDecision.from_mapping(
            decision_value,
            include_description=include_description,
        )
        raw_response = response.get("raw_response")
        if not isinstance(raw_response, str):
            raise ValueError("VLM server response has no raw_response text")
        request_to_decision_ms = (
            float(response["request_to_decision_ms"])
            + (time.perf_counter() - client_parse_started) * 1000
        )
        return (
            decision,
            raw_response,
            float(response["vlm_latency_ms"]),
            request_to_decision_ms,
        )

    def metrics(self) -> dict[str, Any]:
        return self._request({"operation": "metrics"})

    def start_repeat(self, repeat_id: int) -> dict[str, Any]:
        return self._request(
            {"operation": "repeat_start", "repeat_id": repeat_id}
        )

    def end_repeat(self, repeat_id: int) -> dict[str, Any]:
        return self._request(
            {"operation": "repeat_end", "repeat_id": repeat_id}
        )

    def close(self) -> None:
        websocket = getattr(self._client, "_ws", None)
        if websocket is not None:
            websocket.close()
