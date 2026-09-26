"""Bounded tools the local assistant can invoke against one workspace."""

import ast
import base64
import binascii
import ipaddress
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


SKIP_DIRS = {".git", ".venv", "__pycache__", "data", "data_train", "weights", "runs"}
MAX_READ_BYTES = 128_000
MAX_SEARCH_BYTES = 1_000_000
MAX_WRITE_BYTES = 64_000
MAX_SYNTAX_BYTES = 1_000_000
MAX_TEST_OUTPUT = 12_000
MAX_IMAGE_RESPONSE_BYTES = 32_000_000
MAX_IMAGE_BYTES = 20_000_000


TOOL_SPECS = [
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List files and directories directly inside a workspace path.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "Workspace-relative directory; defaults to root."}},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a small UTF-8 text file inside the workspace.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "Workspace-relative file path."}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_text",
            "description": "Search workspace text files for a literal, case-insensitive phrase.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluate a basic arithmetic expression without executing Python.",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write a UTF-8 file inside the workspace. The user must approve every write.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_image",
            "description": "Generate one PNG with a configured local Stable Diffusion WebUI or Forge server. Saving requires user approval.",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {"type": "string"},
                    "negative_prompt": {"type": "string"},
                    "output": {"type": "string", "description": "Workspace-relative .png output path."},
                    "width": {"type": "integer"},
                    "height": {"type": "integer"},
                    "steps": {"type": "integer"},
                },
                "required": ["prompt"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "python_syntax_check",
            "description": "Check a workspace Python file for syntax errors without executing it.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_tests",
            "description": "Run all workspace tests or one test_*.py file. This executes project test code and requires user approval.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Optional test filename such as test_parser.py."},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "git_status",
            "description": "Show short Git status for the selected workspace. Read-only.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "git_diff",
            "description": "Read the tracked working-tree diff for debugging. Output is bounded and the tool is read-only.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]


class WorkspaceTools:
    def __init__(self, root, confirm_write=None, confirm_run=None,
                 image_api_url=None):
        self.root = Path(root).expanduser().resolve()
        self.confirm_write = confirm_write
        self.confirm_run = confirm_run
        self.image_api_url = self._validate_image_api_url(image_api_url)

    @staticmethod
    def _validate_image_api_url(url):
        if url is None or url == "":
            return None
        if not isinstance(url, str):
            raise ValueError("image API URL must be text")
        try:
            url = url.strip()
            parsed = urlsplit(url)
            host = parsed.hostname
            parsed.port
            local_host = (host == "localhost" or
                          (host is not None and ipaddress.ip_address(host).is_loopback))
        except (ValueError, TypeError):
            local_host = False
            parsed = None
        if (parsed is None or parsed.scheme != "http" or not local_host
                or parsed.username or parsed.password
                or parsed.path not in ("", "/") or parsed.query or parsed.fragment):
            raise ValueError(
                "image API must be a local HTTP base URL, such as "
                "http://127.0.0.1:7860")
        return url.rstrip("/")

    def _path(self, relative):
        path = (self.root / relative).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("path must stay inside the workspace")
        return path

    def _visible(self, path):
        return not any(part in SKIP_DIRS for part in path.relative_to(self.root).parts)

    def list_files(self, path="."):
        directory = self._path(path)
        if not directory.is_dir():
            raise ValueError("path is not a directory")
        rows = []
        for child in sorted(directory.iterdir(), key=lambda item: item.name.lower()):
            if child.name.startswith(".") or not self._visible(child):
                continue
            kind = "dir" if child.is_dir() else "file"
            rows.append(f"{kind}\t{child.relative_to(self.root)}")
            if len(rows) == 100:
                break
        return "\n".join(rows) if rows else "(empty)"

    def read_file(self, path):
        file = self._path(path)
        if not file.is_file():
            raise ValueError("path is not a file")
        if file.stat().st_size > MAX_READ_BYTES:
            raise ValueError(f"file exceeds the {MAX_READ_BYTES}-byte read limit")
        return file.read_text(encoding="utf-8", errors="replace")

    def search_text(self, query):
        query = query.strip().casefold()
        if not query:
            raise ValueError("query cannot be empty")
        matches = []
        for directory, subdirs, filenames in os.walk(self.root):
            subdirs[:] = [name for name in subdirs
                          if not name.startswith(".") and name not in SKIP_DIRS
                          and not name.startswith("data_")]
            for filename in filenames:
                if filename.startswith("."):
                    continue
                file = Path(directory) / filename
                try:
                    if file.stat().st_size > MAX_SEARCH_BYTES:
                        continue
                    lines = file.read_text(encoding="utf-8").splitlines()
                except (UnicodeDecodeError, OSError):
                    continue
                for number, line in enumerate(lines, 1):
                    if query in line.casefold():
                        matches.append(f"{file.relative_to(self.root)}:{number}: {line[:300]}")
                        if len(matches) == 50:
                            return "\n".join(matches)
        return "\n".join(matches) if matches else "No matches."

    def write_file(self, path, content):
        file = self._path(path)
        if len(content.encode("utf-8")) > MAX_WRITE_BYTES:
            raise ValueError(f"content exceeds the {MAX_WRITE_BYTES}-byte write limit")
        if self.confirm_write is None or not self.confirm_write(file.relative_to(self.root), file.exists()):
            return "Write denied; the user did not approve it."
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(content, encoding="utf-8")
        return f"Wrote {file.relative_to(self.root)} ({len(content.encode('utf-8'))} bytes)."

    def generate_image(self, prompt, negative_prompt="", output="generated/image.png",
                       width=512, height=512, steps=20):
        if not self.image_api_url:
            return ("Image generation is not configured. Start a local "
                    "Stable Diffusion WebUI or Forge server with its API enabled, "
                    "then set MINI_AGENT_IMAGE_API to its base URL.")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be non-empty text")
        if len(prompt.encode("utf-8")) > 16_000:
            raise ValueError("prompt exceeds 16,000 bytes")
        if not isinstance(negative_prompt, str) or len(negative_prompt.encode("utf-8")) > 8_000:
            raise ValueError("negative_prompt must be text up to 8,000 bytes")
        for name, value in (("width", width), ("height", height)):
            if (not isinstance(value, int) or isinstance(value, bool)
                    or not 64 <= value <= 1024 or value % 8):
                raise ValueError(f"{name} must be a multiple of 8 from 64 to 1024")
        if not isinstance(steps, int) or isinstance(steps, bool) or not 1 <= steps <= 50:
            raise ValueError("steps must be an integer from 1 to 50")
        if not isinstance(output, str) or len(output) > 240:
            raise ValueError("output must be a workspace-relative .png path")
        image_path = self._path(output)
        if image_path.suffix.lower() != ".png":
            raise ValueError("output path must end in .png")
        relative = image_path.relative_to(self.root).as_posix()
        exists = image_path.exists()
        if exists and not image_path.is_file():
            raise ValueError("output path already exists and is not a file")
        if (self.confirm_write is None
                or not self.confirm_write(relative, exists)):
            return "Image generation denied; the user did not approve saving the file."

        payload = json.dumps({
            "prompt": prompt.strip(),
            "negative_prompt": negative_prompt,
            "width": width,
            "height": height,
            "steps": steps,
            "batch_size": 1,
            "save_images": False,
        }).encode("utf-8")
        request = Request(
            f"{self.image_api_url}/sdapi/v1/txt2img",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=600) as response:
                response_data = response.read(MAX_IMAGE_RESPONSE_BYTES + 1)
        except HTTPError as error:
            detail = error.read(1000).decode("utf-8", errors="replace")
            return f"Image API returned HTTP {error.code}: {detail}"
        except (URLError, TimeoutError) as error:
            return f"Could not reach the local image API: {error}"
        if len(response_data) > MAX_IMAGE_RESPONSE_BYTES:
            return "Image API response exceeded the 32 MB limit."
        try:
            result = json.loads(response_data.decode("utf-8"))
            images = result.get("images") if isinstance(result, dict) else None
            encoded = images[0] if isinstance(images, list) and images else None
            if not isinstance(encoded, str):
                raise ValueError("response did not contain an image")
            if encoded.startswith("data:") and "," in encoded:
                encoded = encoded.split(",", 1)[1]
            image_data = base64.b64decode(encoded, validate=True)
        except (UnicodeDecodeError, json.JSONDecodeError, binascii.Error,
                IndexError, ValueError) as error:
            return f"Image API returned invalid image data: {error}"
        if len(image_data) > MAX_IMAGE_BYTES:
            return "Generated image exceeded the 20 MB limit."
        if not image_data.startswith(b"\x89PNG\r\n\x1a\n"):
            return "Image API did not return a PNG image."
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image_path.write_bytes(image_data)
        return f"Saved generated image to {relative} ({len(image_data)} bytes)."

    def python_syntax_check(self, path):
        file = self._path(path)
        if not file.is_file() or file.suffix != ".py":
            raise ValueError("path must be a Python source file")
        if file.stat().st_size > MAX_SYNTAX_BYTES:
            raise ValueError(f"file exceeds the {MAX_SYNTAX_BYTES}-byte syntax-check limit")
        source = file.read_text(encoding="utf-8")
        try:
            compile(source, str(file.relative_to(self.root)), "exec")
        except SyntaxError as error:
            return (f"SyntaxError in {file.relative_to(self.root)} at line "
                    f"{error.lineno}: {error.msg}")
        return f"Syntax OK: {file.relative_to(self.root)}"

    def run_tests(self, pattern=None):
        tests_dir = self.root / "tests"
        if not tests_dir.is_dir():
            return "Test error: workspace has no tests/ directory."
        if pattern is not None:
            if not isinstance(pattern, str) or not re.fullmatch(
                    r"test_[A-Za-z0-9_]+\.py", pattern):
                raise ValueError("pattern must be a test_*.py filename")
            target = tests_dir / pattern
            if (not target.is_file()
                    or target.resolve().parent != tests_dir.resolve()):
                return f"Test error: no in-workspace test file named {pattern}."
        if self.confirm_run is None or not self.confirm_run():
            return "Test run denied; user approval is required."
        command = [sys.executable, "-m", "unittest", "discover", "-s", "tests"]
        if pattern is not None:
            command.extend(("-p", pattern))
        command.append("-v")
        try:
            result = subprocess.run(command, cwd=self.root, text=True,
                                    capture_output=True, timeout=180, check=False)
        except subprocess.TimeoutExpired:
            return "Test run timed out after 180 seconds."
        except OSError as error:
            return f"Test error: could not start Python: {error}"
        output = (result.stdout + result.stderr).strip()
        if len(output) > MAX_TEST_OUTPUT:
            output = output[-MAX_TEST_OUTPUT:]
        return f"Tests exited with status {result.returncode}.\n{output}"

    def git_status(self):
        return self._git_output(["status", "--short", "--untracked-files=normal"])

    def git_diff(self):
        return self._git_output(["diff", "--no-ext-diff", "--unified=3"])

    def _git_output(self, arguments):
        try:
            result = subprocess.run(["git", *arguments], cwd=self.root,
                                    text=True, capture_output=True, timeout=15,
                                    check=False)
        except (OSError, subprocess.TimeoutExpired) as error:
            return f"Git error: {error}"
        if result.returncode:
            return f"Git error: {(result.stderr or result.stdout).strip()}"
        output = (result.stdout + result.stderr).strip()
        if not output:
            return "(no changes)"
        if len(output) > MAX_TEST_OUTPUT:
            output = output[-MAX_TEST_OUTPUT:]
        return output

    def call(self, name, arguments):
        methods = {
            "list_files": self.list_files,
            "read_file": self.read_file,
            "search_text": self.search_text,
            "calculate": calculate,
            "write_file": self.write_file,
            "generate_image": self.generate_image,
            "python_syntax_check": self.python_syntax_check,
            "run_tests": self.run_tests,
            "git_status": self.git_status,
            "git_diff": self.git_diff,
        }
        if name not in methods:
            return f"Tool error: unknown tool {name!r}."
        try:
            return str(methods[name](**arguments))
        except (ArithmeticError, OSError, SyntaxError, TypeError, ValueError) as error:
            return f"Tool error: {error}"


def calculate(expression):
    if not isinstance(expression, str) or len(expression) > 256:
        raise ValueError("expression must be text no longer than 256 characters")
    tree = ast.parse(expression, mode="eval")
    binary = {
        ast.Add: lambda a, b: a + b,
        ast.Sub: lambda a, b: a - b,
        ast.Mult: lambda a, b: a * b,
        ast.Div: lambda a, b: a / b,
        ast.FloorDiv: lambda a, b: a // b,
        ast.Mod: lambda a, b: a % b,
        ast.Pow: lambda a, b: a ** b,
    }

    def evaluate(node):
        if isinstance(node, ast.Expression):
            return evaluate(node.body)
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return node.value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = evaluate(node.operand)
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.BinOp) and type(node.op) in binary:
            left, right = evaluate(node.left), evaluate(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 12:
                raise ValueError("exponent magnitude is limited to 12")
            result = binary[type(node.op)](left, right)
            if isinstance(result, int) and result.bit_length() > 256:
                raise ValueError("integer result is too large")
            if isinstance(result, float) and not math.isfinite(result):
                raise ValueError("result is not finite")
            return result
        raise ValueError("only numeric constants and basic arithmetic are allowed")

    return str(evaluate(tree))
