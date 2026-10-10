"""Local chat protocol tests: no network calls or Ollama installation needed."""
import io
import json
import unittest
from unittest.mock import patch

from scripts.greenlight_next_chat import ollama_reply


class OllamaReplyTests(unittest.TestCase):
    def test_sends_messages_to_loopback_only(self):
        observed = {}

        def fake_open(request, timeout):
            observed["url"] = request.full_url
            observed["body"] = json.loads(request.data)
            return io.BytesIO(b'{"message":{"content":"Hello!"}}')

        with patch("urllib.request.urlopen", side_effect=fake_open):
            answer = ollama_reply("test-model", [{"role": "user", "content": "hi"}])
        self.assertEqual(answer, "Hello!")
        self.assertEqual(observed["url"], "http://127.0.0.1:11434/api/chat")
        self.assertEqual(observed["body"]["model"], "test-model")
        self.assertEqual(observed["body"]["messages"][0]["content"], "hi")
        self.assertFalse(observed["body"]["stream"])

    def test_missing_reply_is_error(self):
        with patch("urllib.request.urlopen", return_value=io.BytesIO(b'{}')):
            with self.assertRaises(ValueError):
                ollama_reply("test-model", [])


if __name__ == "__main__":
    unittest.main()
