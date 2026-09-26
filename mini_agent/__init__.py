"""Local model-backed workspace assistant."""

from .agent import LocalAgent
from .ollama import OllamaClient

__all__ = ["LocalAgent", "OllamaClient"]
