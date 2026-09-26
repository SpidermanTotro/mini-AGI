import base64
import io
import json
import tempfile
import subprocess
import unittest
from pathlib import Path
from unittest.mock import patch

from mini_agent.agent import LocalAgent
from mini_agent.tools import WorkspaceTools, calculate


class FakeClient:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.requests = []

    def chat(self, messages, tools):
        self.requests.append((messages, tools))
        return next(self.replies)


class LocalAgentTests(unittest.TestCase):
    def test_calculator_is_arithmetic_only(self):
        self.assertEqual(calculate("(12 + 8) * 3"), "60")
        with self.assertRaises(ValueError):
            calculate("__import__('os').getcwd()")
        with self.assertRaises(ValueError):
            calculate("2 ** 1000")
        tools = WorkspaceTools(".")
        self.assertIn("division by zero", tools.call(
            "calculate", {"expression": "1 / 0"}))
        self.assertIn("invalid syntax", tools.call(
            "calculate", {"expression": "2 +"}))

    def test_workspace_paths_are_confined_and_writes_need_approval(self):
        with tempfile.TemporaryDirectory() as root, tempfile.TemporaryDirectory() as outside:
            root_path = Path(root)
            (root_path / "note.txt").write_text("local note", encoding="utf-8")
            tools = WorkspaceTools(root)
            self.assertEqual(tools.call("read_file", {"path": "note.txt"}), "local note")
            self.assertIn("path must stay inside", tools.call(
                "read_file", {"path": str(Path(outside) / "secret.txt")}))
            denied = tools.call("write_file", {"path": "new.txt", "content": "no"})
            self.assertIn("denied", denied)
            self.assertFalse((root_path / "new.txt").exists())

            allowed = WorkspaceTools(root, confirm_write=lambda path, exists: True)
            result = allowed.call("write_file", {"path": "new.txt", "content": "yes"})
            self.assertIn("Wrote new.txt", result)
            self.assertEqual((root_path / "new.txt").read_text(encoding="utf-8"), "yes")

    def test_image_generation_is_local_and_requires_approval(self):
        with tempfile.TemporaryDirectory() as root:
            unconfigured = WorkspaceTools(root)
            self.assertIn("not configured", unconfigured.call(
                "generate_image", {"prompt": "A red fox"}))
            with self.assertRaises(ValueError):
                WorkspaceTools(root, image_api_url="http://example.com")

            denied = WorkspaceTools(root, image_api_url="http://127.0.0.1:7860")
            with patch("mini_agent.tools.urlopen") as open_api:
                result = denied.call("generate_image", {"prompt": "A red fox"})
            self.assertIn("denied", result)
            open_api.assert_not_called()

    def test_image_generation_saves_validated_png_inside_workspace(self):
        png = b"\x89PNG\r\n\x1a\nimage-data"
        response = json.dumps({
            "images": [base64.b64encode(png).decode("ascii")],
        }).encode("utf-8")
        with tempfile.TemporaryDirectory() as root:
            tools = WorkspaceTools(
                root,
                confirm_write=lambda path, exists: path == "generated/red-fox.png" and not exists,
                image_api_url="http://127.0.0.1:7860",
            )
            with patch("mini_agent.tools.urlopen", return_value=io.BytesIO(response)) as open_api:
                result = tools.call("generate_image", {
                    "prompt": "A red fox",
                    "output": "generated/red-fox.png",
                    "width": 256,
                    "height": 384,
                    "steps": 12,
                })
            self.assertIn("Saved generated image", result)
            output = Path(root) / "generated" / "red-fox.png"
            self.assertEqual(output.read_bytes(), png)
            request = open_api.call_args.args[0]
            self.assertEqual(request.full_url,
                             "http://127.0.0.1:7860/sdapi/v1/txt2img")
            payload = json.loads(request.data)
            self.assertEqual(payload["prompt"], "A red fox")
            self.assertEqual(payload["width"], 256)
            self.assertEqual(payload["height"], 384)
            self.assertEqual(payload["steps"], 12)

    def test_image_generation_rejects_unsafe_arguments(self):
        with tempfile.TemporaryDirectory() as root:
            tools = WorkspaceTools(root, image_api_url="http://localhost:7860")
            self.assertIn("multiple of 8", tools.call("generate_image", {
                "prompt": "A red fox", "width": 257,
            }))
            self.assertIn("path must stay inside", tools.call("generate_image", {
                "prompt": "A red fox", "output": "../outside.png",
            }))
            with self.assertRaises(ValueError):
                WorkspaceTools(root, image_api_url="http://127.0.0.1:bad")

    def test_python_syntax_check_does_not_execute_source(self):
        with tempfile.TemporaryDirectory() as root:
            root_path = Path(root)
            (root_path / "valid.py").write_text("value = 42\n", encoding="utf-8")
            (root_path / "broken.py").write_text("def missing(:\n", encoding="utf-8")
            tools = WorkspaceTools(root)
            self.assertEqual(tools.call(
                "python_syntax_check", {"path": "valid.py"}),
                "Syntax OK: valid.py")
            self.assertIn("SyntaxError in broken.py at line 1", tools.call(
                "python_syntax_check", {"path": "broken.py"}))

    def test_test_runner_requires_approval_and_uses_fixed_command(self):
        with tempfile.TemporaryDirectory() as root:
            (Path(root) / "tests").mkdir()
            denied = WorkspaceTools(root)
            self.assertIn("approval is required", denied.call("run_tests", {}))

            allowed = WorkspaceTools(root, confirm_run=lambda: True)
            completed = __import__("subprocess").CompletedProcess(
                args=[], returncode=0, stdout="Ran 0 tests", stderr="")
            with patch("mini_agent.tools.subprocess.run", return_value=completed) as run:
                result = allowed.call("run_tests", {})
            self.assertIn("status 0", result)
            command = run.call_args.args[0]
            self.assertEqual(command[1:5], ["-m", "unittest", "discover", "-s"])
            self.assertNotIn("shell", run.call_args.kwargs)

    def test_git_diagnostics_use_fixed_read_only_commands(self):
        with tempfile.TemporaryDirectory() as root:
            tools = WorkspaceTools(root)
            status = subprocess.CompletedProcess([], 0, " M file.py\n", "")
            diff = subprocess.CompletedProcess([], 0, "x" * 20_000, "")
            with patch("mini_agent.tools.subprocess.run", side_effect=(status, diff)) as run:
                self.assertIn("M file.py", tools.call("git_status", {}))
                self.assertEqual(len(tools.call("git_diff", {})), 12_000)
            commands = [call.args[0] for call in run.call_args_list]
            self.assertEqual(commands[0], ["git", "status", "--short", "--untracked-files=normal"])
            self.assertEqual(commands[1][:3], ["git", "diff", "--no-ext-diff"])
            self.assertTrue(all("shell" not in call.kwargs for call in run.call_args_list))

    def test_tool_loop_and_sqlite_history_survive_restart(self):
        with tempfile.TemporaryDirectory() as root:
            database = Path(root) / "history.sqlite3"
            tool_reply = {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": "call-1",
                    "function": {
                        "name": "calculate",
                        "arguments": {"expression": "6 * 7"},
                    },
                }],
            }
            client = FakeClient([tool_reply, {"role": "assistant", "content": "42"}])
            agent = LocalAgent(client, root, database)
            self.assertEqual(agent.run("What is six times seven?"), "42")
            agent.close()

            resumed_client = FakeClient([{"role": "assistant", "content": "I remember."}])
            resumed = LocalAgent(resumed_client, root, database)
            self.assertEqual(resumed.run("Do you remember?"), "I remember.")
            history = resumed_client.requests[0][0]
            self.assertEqual(history[1]["content"], "What is six times seven?")
            self.assertEqual(history[3]["role"], "tool")
            self.assertEqual(history[3]["content"], "42")
            resumed.close()

    def test_empty_user_message_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            agent = LocalAgent(FakeClient([]), root, ":memory:")
            with self.assertRaises(ValueError):
                agent.run("  ")
            agent.close()


if __name__ == "__main__":
    unittest.main()
