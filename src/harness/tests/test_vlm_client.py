"""Unit tests for the Agentic OpenAI-compatible vLLM client."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from harness.vlm_client import AgenticVLMClient
from harness.vlm_client import _openai_messages
from harness.vlm_client import _parse_json_object


class VLMClientMessageTest(unittest.TestCase):
    def test_interleaves_local_images_with_text(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first image.png"
            second = Path(directory) / "second.png"
            first.write_bytes(b"first")
            second.write_bytes(b"second")

            converted = _openai_messages(
                [
                    {"role": "system", "content": "Return JSON."},
                    {
                        "role": "user",
                        "content": "Before <image> between <image> after",
                    },
                ],
                [str(first), str(second)],
            )

        self.assertEqual(converted[0]["content"], "Return JSON.")
        parts = converted[1]["content"]
        self.assertEqual(
            [part["type"] for part in parts],
            ["text", "image_url", "text", "image_url", "text"],
        )
        self.assertEqual(parts[1]["image_url"]["url"], first.resolve().as_uri())
        self.assertEqual(parts[3]["image_url"]["url"], second.resolve().as_uri())

    def test_rejects_marker_and_path_count_mismatch(self) -> None:
        with self.assertRaisesRegex(ValueError, "more image markers"):
            _openai_messages(
                [{"role": "user", "content": "<image>"}],
                [],
            )

    def test_parses_json_after_wrapper_text(self) -> None:
        self.assertEqual(
            _parse_json_object('answer follows: {"status": "ok"}\nignored'),
            {"status": "ok"},
        )

    def test_client_sends_structured_openai_request(self) -> None:
        with mock.patch.object(
            AgenticVLMClient,
            "_request_json",
            side_effect=[
                {"data": [{"id": "qwen-test"}]},
                {
                    "choices": [
                        {"message": {"content": '{"status":"completed"}'}}
                    ]
                },
            ],
        ) as request_json:
            client = AgenticVLMClient(
                "127.0.0.1",
                8000,
                model_name="qwen-test",
                model_path="/models/qwen-test",
                max_tokens=512,
            )
            value, raw, _, _ = client.infer_json(
                messages=[{"role": "user", "content": "Return JSON."}],
                planning_phase="init_planning",
            )

        self.assertEqual(value, {"status": "completed"})
        self.assertEqual(raw, '{"status":"completed"}')
        payload = request_json.call_args_list[1].args[2]
        self.assertEqual(payload["model"], "qwen-test")
        self.assertEqual(payload["max_tokens"], 512)
        self.assertEqual(payload["response_format"], {"type": "json_object"})
        self.assertEqual(client.metrics()["planning_phases"]["init_planning"]["vlm_inference"]["steps"], 1)


if __name__ == "__main__":
    unittest.main()
