"""OpenAI-compatible client for the Agentic vLLM service."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from oracle_plus.runtime_metrics import latency_summary


PLANNING_PHASES = ("init_planning", "in_task_planning", "auxiliary")


def _parse_json_object(text: str) -> dict[str, Any]:
    """Parse the first complete JSON object from one model response."""

    object_start = text.find("{")
    if object_start < 0:
        raise ValueError("vLLM response contains no JSON object")
    try:
        value, _ = json.JSONDecoder().raw_decode(text[object_start:])
    except json.JSONDecodeError as error:
        raise ValueError("vLLM response contains invalid JSON") from error
    if not isinstance(value, dict):
        raise ValueError("vLLM response JSON must be an object")
    return value


def _openai_messages(
    messages: list[dict[str, str]],
    images: list[str],
) -> list[dict[str, Any]]:
    """Replace ordered ``<image>`` markers with local OpenAI image parts."""

    image_index = 0
    converted: list[dict[str, Any]] = []
    for message in messages:
        role = message.get("role")
        content = message.get("content")
        if not isinstance(role, str) or not isinstance(content, str):
            raise TypeError("every VLM message requires string role and content")
        pieces = content.split("<image>")
        if len(pieces) == 1:
            converted.append({"role": role, "content": content})
            continue
        parts: list[dict[str, Any]] = []
        for index, piece in enumerate(pieces):
            if piece:
                parts.append({"type": "text", "text": piece})
            if index == len(pieces) - 1:
                continue
            if image_index >= len(images):
                raise ValueError("VLM prompt contains more image markers than paths")
            image_path = Path(images[image_index]).expanduser().resolve()
            if not image_path.is_file():
                raise FileNotFoundError(f"VLM image does not exist: {image_path}")
            parts.append(
                {
                    "type": "image_url",
                    "image_url": {"url": image_path.as_uri()},
                }
            )
            image_index += 1
        converted.append({"role": role, "content": parts})
    if image_index != len(images):
        raise ValueError("VLM request has more image paths than prompt markers")
    return converted


class AgenticVLMClient:
    """Call one locally served vLLM model and retain per-phase timings."""

    def __init__(
        self,
        host: str,
        port: int,
        *,
        model_name: str,
        model_path: str,
        max_tokens: int,
        timeout_seconds: float = 600,
    ) -> None:
        self.host = host
        self.port = port
        self.model_name = model_name
        self.model_path = str(Path(model_path).expanduser().resolve())
        self.max_tokens = max_tokens
        self.timeout_seconds = timeout_seconds
        self.base_url = f"http://{host}:{port}"
        self._vlm_times_ms: list[float] = []
        self._request_times_ms: list[float] = []
        self._phase_vlm_times_ms = {phase: [] for phase in PLANNING_PHASES}
        self._phase_request_times_ms = {phase: [] for phase in PLANNING_PHASES}
        models = self._request_json("GET", "/v1/models")
        served_models = {
            item.get("id")
            for item in models.get("data", [])
            if isinstance(item, dict)
        }
        if model_name not in served_models:
            raise RuntimeError(
                f"vLLM does not serve {model_name!r}; available={sorted(served_models)}"
            )
        self.server_metadata = {
            "protocol": "agentic_vllm_openai_v1",
            "model_name": model_name,
            "model_path": self.model_path,
            "max_tokens": max_tokens,
            "endpoint": self.base_url,
        }

    def _request_json(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        body = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=body,
            headers=headers,
            method=method,
        )
        try:
            with urllib.request.urlopen(
                request,
                timeout=self.timeout_seconds,
            ) as response:
                value = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"vLLM request failed with HTTP {error.code}: {detail}"
            ) from error
        except urllib.error.URLError as error:
            raise ConnectionError(
                f"vLLM is unavailable at {self.base_url}: {error.reason}"
            ) from error
        if not isinstance(value, dict):
            raise ValueError("vLLM HTTP response must be a JSON object")
        return value

    def infer_json(
        self,
        *,
        messages: list[dict[str, str]],
        images: list[str] | None = None,
        planning_phase: str,
    ) -> tuple[dict[str, Any], str, float, float]:
        if planning_phase not in PLANNING_PHASES:
            raise ValueError(
                f"planning_phase must be one of {PLANNING_PHASES}, got "
                f"{planning_phase!r}"
            )
        request_started = time.perf_counter()
        response = self._request_json(
            "POST",
            "/v1/chat/completions",
            {
                "model": self.model_name,
                "messages": _openai_messages(messages, list(images or [])),
                "temperature": 0.0,
                "top_p": 1.0,
                "max_tokens": self.max_tokens,
                "seed": 42,
                "response_format": {"type": "json_object"},
            },
        )
        response_received = time.perf_counter()
        try:
            raw_response = response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as error:
            raise ValueError("vLLM response has no assistant content") from error
        if not isinstance(raw_response, str):
            raise ValueError("vLLM assistant content must be text")
        value = _parse_json_object(raw_response)
        request_to_json_ms = (time.perf_counter() - request_started) * 1000
        vlm_latency_ms = (response_received - request_started) * 1000
        self._vlm_times_ms.append(vlm_latency_ms)
        self._request_times_ms.append(request_to_json_ms)
        self._phase_vlm_times_ms[planning_phase].append(vlm_latency_ms)
        self._phase_request_times_ms[planning_phase].append(request_to_json_ms)
        return value, raw_response, vlm_latency_ms, request_to_json_ms

    def metrics(self) -> dict[str, Any]:
        return {
            **self.server_metadata,
            "vlm_single_call": latency_summary(self._vlm_times_ms),
            "request_to_json": latency_summary(self._request_times_ms),
            "planning_phases": {
                phase: {
                    "vlm_inference": latency_summary(
                        self._phase_vlm_times_ms[phase]
                    ),
                    "request_to_json": latency_summary(
                        self._phase_request_times_ms[phase]
                    ),
                }
                for phase in PLANNING_PHASES
            },
        }

    def close(self) -> None:
        """The urllib client does not retain an open connection."""
