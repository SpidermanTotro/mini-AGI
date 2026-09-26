"""Small HTTP client for Ollama's local chat and tool-calling API."""

import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class OllamaClient:
    def __init__(self, base_url="http://127.0.0.1:11434", model="qwen3:8b", timeout=180):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def chat(self, messages, tools):
        payload = json.dumps({
            "model": self.model,
            "messages": messages,
            "tools": tools,
            "stream": False,
        }).encode("utf-8")
        request = Request(
            f"{self.base_url}/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")[:1000]
            raise RuntimeError(f"Ollama returned HTTP {error.code}: {detail}") from error
        except URLError as error:
            raise RuntimeError(
                f"Cannot reach Ollama at {self.base_url}; start Ollama and verify the local URL."
            ) from error
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise RuntimeError("Ollama returned an invalid JSON response") from error

        message = result.get("message")
        if not isinstance(message, dict) or not isinstance(message.get("role"), str):
            raise RuntimeError("Ollama response did not contain a chat message")
        return message
