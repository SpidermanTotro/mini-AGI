"""Permission-limited local workspace reader for Greenlight Next.

Read-only by design. No shell, network, writes, or arbitrary path access.
"""
from __future__ import annotations

from pathlib import Path


class WorkspaceReader:
    def __init__(self, root: str | Path, max_bytes: int = 65536):
        self.root = Path(root).resolve(strict=True)
        if not self.root.is_dir():
            raise ValueError("workspace root must be a directory")
        if max_bytes < 1:
            raise ValueError("max_bytes must be positive")
        self.max_bytes = max_bytes

    def _resolve(self, relative_path: str) -> Path:
        path = Path(relative_path)
        if path.is_absolute() or not relative_path or ".." in path.parts:
            raise PermissionError("path must be relative to the workspace")
        target = (self.root / path).resolve(strict=True)
        if not target.is_relative_to(self.root):
            raise PermissionError("path escapes the workspace")
        return target

    def read_text(self, relative_path: str) -> str:
        target = self._resolve(relative_path)
        if not target.is_file():
            raise ValueError("target is not a regular file")
        if target.stat().st_size > self.max_bytes:
            raise ValueError("file exceeds configured read limit")
        with target.open("rb") as handle:
            data = handle.read(self.max_bytes + 1)
        if len(data) > self.max_bytes:
            raise ValueError("file exceeds configured read limit")
        return data.decode("utf-8")
