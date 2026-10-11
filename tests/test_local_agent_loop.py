import tempfile
import unittest
from pathlib import Path

from minagi.local_agent_loop import run_local_agent
from minagi.local_assistant_tools import LocalAssistantTools
from minagi.local_tool_dispatcher import LocalToolDispatcher


class AgentLoopTests(unittest.TestCase):
    def test_read_then_answer(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "note.txt").write_text("dragon", encoding="utf-8")
            dispatcher = LocalToolDispatcher(LocalAssistantTools(
                root, root / "history.db", allow_workspace_reads=True))
            seen = []

            def model(messages, tools):
                seen.append(messages)
                if len(seen) == 1:
                    return {"role": "assistant", "content": "",
                            "tool_calls": [{"function": {
                                "name": "read_workspace_file",
                                "arguments": {"relative_path": "note.txt"}}}]}
                return {"role": "assistant", "content": "Read complete"}

            answer, transcript = run_local_agent(
                model, [{"role": "user", "content": "read note"}], dispatcher)
            self.assertEqual(answer, "Read complete")
            self.assertIn('"dragon"', transcript[-2]["content"])
            self.assertEqual(transcript[-2]["role"], "tool")

    def test_tool_budget_stops_repeated_requests(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dispatcher = LocalToolDispatcher(LocalAssistantTools(root, root / "history.db"))

            def model(messages, tools):
                return {"role": "assistant", "content": "",
                        "tool_calls": [{"function": {
                            "name": "read_workspace_file",
                            "arguments": {"relative_path": "missing"}}}]}

            with self.assertRaises(RuntimeError):
                run_local_agent(model, [{"role": "user", "content": "go"}],
                                dispatcher, max_tool_calls=1)

    def test_malformed_reply_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dispatcher = LocalToolDispatcher(LocalAssistantTools(root, root / "history.db"))
            with self.assertRaises(ValueError):
                run_local_agent(lambda messages, tools: {"role": "user"},
                                [], dispatcher)


if __name__ == "__main__":
    unittest.main()
