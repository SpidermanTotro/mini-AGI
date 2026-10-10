"""Opt-in local terminal chat using Ollama on loopback only.

No cloud service, shell execution, or checkpoint mutation. This uses an
installed Ollama model, not Greenlight's own checkpoint inference.
"""
from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request
from pathlib import Path

from minagi.conversation_store import ConversationStore


def ollama_reply(model: str, messages: list[dict], timeout: float = 120) -> str:
    payload = json.dumps({"model": model, "messages": messages,
                          "stream": False}).encode("utf-8")
    request = urllib.request.Request(
        "http://127.0.0.1:11434/api/chat", data=payload,
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        result = json.load(response)
    answer = result.get("message", {}).get("content")
    if not isinstance(answer, str):
        raise ValueError("local Ollama returned no text response")
    return answer


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Greenlight Next local Ollama chat")
    parser.add_argument("--model", default="qwen3:8b")
    parser.add_argument("--db", required=True, help="Opt-in SQLite history path")
    parser.add_argument("--conversation", default="default")
    args = parser.parse_args(argv)
    print("Greenlight Next local chat (Ollama 127.0.0.1 only). /quit to exit.")
    with ConversationStore(Path(args.db)) as history:
        while True:
            try:
                prompt = input("you> ")
            except (EOFError, KeyboardInterrupt):
                print()
                return 0
            if prompt.strip() == "/quit":
                return 0
            if not prompt.strip():
                continue
            prior = history.history(args.conversation)
            messages = [{"role": item["role"], "content": item["content"]}
                        for item in prior if item["role"] in ("user", "assistant")]
            messages.append({"role": "user", "content": prompt})
            try:
                reply = ollama_reply(args.model, messages)
            except (urllib.error.URLError, TimeoutError, ValueError,
                    json.JSONDecodeError) as exc:
                print(f"Local inference failed (conversation unchanged): {exc}")
                continue
            history.add(args.conversation, "user", prompt)
            history.add(args.conversation, "assistant", reply)
            print(f"assistant> {reply}")


if __name__ == "__main__":
    raise SystemExit(main())
