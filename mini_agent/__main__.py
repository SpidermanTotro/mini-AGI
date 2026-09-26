"""Run the local assistant with ``python -m mini_agent``."""

import argparse
import os
from pathlib import Path

from .agent import LocalAgent
from .ollama import OllamaClient


def approve_write(path, replacing):
    action = "replace" if replacing else "create"
    answer = input(f"\nAllow the assistant to {action} {path}? [y/N] ")
    return answer.strip().casefold() in {"y", "yes"}


def approve_test_run():
    answer = input(
        "\nRun the workspace unittest suite? This executes repository test code. [y/N] "
    )
    return answer.strip().casefold() in {"y", "yes"}


def main():
    parser = argparse.ArgumentParser(description="Local Ollama assistant with guarded workspace tools")
    parser.add_argument("--workspace", default=".", help="workspace root for read/write tools")
    parser.add_argument("--model", default=os.environ.get("OLLAMA_MODEL", "qwen3:8b"))
    parser.add_argument("--url", default=os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434"))
    parser.add_argument(
        "--database",
        default=os.environ.get("MINI_AGENT_DB", "~/.local/share/mini-agi/agent.sqlite3"),
        help="SQLite conversation history path",
    )
    parser.add_argument("--conversation", default="default")
    parser.add_argument("--reset", action="store_true", help="clear this conversation before starting")
    args = parser.parse_args()

    workspace = Path(args.workspace).expanduser().resolve()
    if not workspace.is_dir():
        parser.error(f"workspace is not a directory: {workspace}")

    agent = LocalAgent(
        OllamaClient(base_url=args.url, model=args.model),
        workspace=workspace,
        database=args.database,
        confirm_write=approve_write,
        confirm_run=approve_test_run,
    )
    if args.reset:
        agent.reset(args.conversation)

    print(f"Local assistant | model: {args.model} | workspace: {workspace}")
    print("Type /reset to clear this conversation or /exit to quit.")
    try:
        while True:
            try:
                text = input("\nyou> ").strip()
            except EOFError:
                break
            if not text:
                continue
            if text == "/exit":
                break
            if text == "/reset":
                agent.reset(args.conversation)
                print("Conversation cleared.")
                continue
            try:
                response = agent.run(text, conversation=args.conversation)
                print(f"\nassistant> {response}")
            except (RuntimeError, ValueError) as error:
                print(f"\nagent error> {error}")
    except KeyboardInterrupt:
        print()
    finally:
        agent.close()


if __name__ == "__main__":
    main()
