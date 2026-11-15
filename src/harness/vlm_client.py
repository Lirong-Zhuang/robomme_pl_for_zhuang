"""Client for the Qwen3.8 Agentic JSON-planning service."""

from __future__ import annotations

import socket
import time
from typing import Any

from openpi_client.websocket_client_policy import WebsocketClientPolicy
from websockets.exceptions import ConnectionClosed


class AgenticVLMClient:
    def __init__(self, host: str, port: int) -> None:
        self.host = host
        self.port = port
        self._client = self._connect()
        self.server_metadata = self._client.get_server_metadata()

    def _connect(self) -> WebsocketClientPolicy:
        return WebsocketClientPolicy(host=self.host, port=self.port)

    def _request(self, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            response = self._client.infer(payload)
        except ConnectionClosed:
            try:
                with socket.create_connection((self.host, self.port), timeout=2):
                    pass
            except OSError as error:
                raise ConnectionError(
                    f"Agentic VLM server is not listening at {self.host}:{self.port}"
                ) from error
            self._client = self._connect()
            self.server_metadata = self._client.get_server_metadata()
            response = self._client.infer(payload)
        if not isinstance(response, dict):
            raise ValueError("Agentic VLM response must be an object")
        return response

    def infer_json(
        self,
        *,
        messages: list[dict[str, str]],
        images: list[str] | None = None,
        planning_phase: str,
    ) -> tuple[dict[str, Any], str, float, float]:
        response = self._request(
            {
                "operation": "infer_json",
                "messages": messages,
                "images": list(images or []),
                "planning_phase": planning_phase,
            }
        )
        parse_started = time.perf_counter()
        value = response.get("json")
        if not isinstance(value, dict):
            raise ValueError("Agentic VLM response has no JSON object")
        raw_response = response.get("raw_response")
        if not isinstance(raw_response, str):
            raise ValueError("Agentic VLM response has no raw_response")
        request_to_json_ms = float(response["request_to_json_ms"])
        request_to_json_ms += (time.perf_counter() - parse_started) * 1000
        return (
            value,
            raw_response,
            float(response["vlm_latency_ms"]),
            request_to_json_ms,
        )

    def metrics(self) -> dict[str, Any]:
        return self._request({"operation": "metrics"})

    def close(self) -> None:
        websocket = getattr(self._client, "_ws", None)
        if websocket is not None:
            websocket.close()
