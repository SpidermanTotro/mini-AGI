"""Minimal local terminal chat with explicit opt-in history.

Uses the existing Greenlight Ollama chat backend through its documented CLI.
Does not claim to load a Greenlight checkpoint or call cloud APIs.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from minagi.conversation_store import ConversationStore


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Greenlight Next local chat")
    parser.add_argument("--model", default="qwen3:8b")
    parser.add_argument("--workspace", default=".")
    parser.add_argument("--db", required=True, help="Opt-in SQLite history path")
    parser.add_argument("--conversation", default="default")
    args = parser.parse_args(argv)
    workspace = Path(args.workspace).resolve(strict=True)
    if not workspace.is_dir():
        parser.error("workspace must be a directory")
    print("Greenlight Next local chat (Ollama backend). /quit to exit.")
    with ConversationStore(args.db) as history:
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
            # Existing chat CLI handles inference; no shell or remote execution.
            # History is stored separately until a verified context-injection
            # integration is implemented.
            history.add(args.conversation, "user", prompt)
            command = [sys.executable, "greenlight.py", "chat",
                       "--model", args.model, "--workspace", str(workspace)]
            print("This prototype stores history only. Use the existing chat "
                  "command for model interaction:")
            print(" ".join(command))
            print("No model invocation performed in this prototype.")


if __name__ == "__main__":
    raise SystemExit(main())
