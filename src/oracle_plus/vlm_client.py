"""WebSocket client for the Qwen3.8 Oracle Plus inference service."""

from __future__ import annotations

from typing import Any

from openpi_client.websocket_client_policy import WebsocketClientPolicy

from .vlm import SubgoalDecision


class VLMSubgoalPlannerClient:
    """Expose the remote 27B service through the local planner interface."""

    def __init__(self, host: str, port: int) -> None:
        self._client = WebsocketClientPolicy(host=host, port=port)
        self.server_metadata = self._client.get_server_metadata()

    def infer(
        self,
        *,
        messages: list[dict[str, str]],
        images: list[str],
        include_description: bool,
    ) -> tuple[SubgoalDecision, str, float, float]:
        response = self._client.infer(
            {
                "operation": "infer",
                "messages": messages,
                "images": images,
                "include_description": include_description,
            }
        )
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
        return (
            decision,
            raw_response,
            float(response["vlm_latency_ms"]),
            float(response["request_to_decision_ms"]),
        )

    def metrics(self) -> dict[str, Any]:
        response = self._client.infer({"operation": "metrics"})
        if not isinstance(response, dict):
            raise ValueError("VLM server metrics response is not an object")
        return response

    def close(self) -> None:
        websocket = getattr(self._client, "_ws", None)
        if websocket is not None:
            websocket.close()
