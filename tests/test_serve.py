import importlib.util
import unittest
from types import SimpleNamespace
from unittest.mock import ANY, patch


FLASK_AVAILABLE = importlib.util.find_spec("flask") is not None


@unittest.skipUnless(FLASK_AVAILABLE, "Flask is an optional dependency")
class ServeApiTests(unittest.TestCase):
    def setUp(self):
        import serve
        self.serve = serve

    def test_chat_rejects_malformed_request_shapes(self):
        with self.serve.app.test_client() as client:
            for payload in (
                    ["not", "an", "object"],
                    {"messages": "not a list"},
                    {"messages": [{"role": "user", "content": None}]},
                    {"messages": [{"role": "system", "content": "x"}]}):
                with self.subTest(payload=payload):
                    response = client.post("/api/chat", json=payload)
                    self.assertEqual(response.status_code, 400)

    def test_chat_rejects_unbounded_generation_requests(self):
        with self.serve.app.test_client() as client:
            for value in (0, -1, 4097, True, "100"):
                with self.subTest(value=value):
                    response = client.post("/api/chat", json={
                        "messages": [], "max_new": value,
                    })
                    self.assertEqual(response.status_code, 400)

    def test_chat_accepts_browser_message_roles(self):
        state = {"model": SimpleNamespace(
            cfg=SimpleNamespace(block=1024)), "tok": None}
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "bot", "content": "hi"},
        ]
        with patch.dict(self.serve.STATE, state), \
                patch("serve.stream", return_value=[{"t": "o"}]) as stream, \
                patch("serve.remember", return_value=None), \
                patch("serve.learn_state", return_value=None), \
                patch("serve.resident_experts", return_value=None), \
                self.serve.app.test_client() as client:
            response = client.post("/api/chat", json={
                "messages": messages, "max_new": 400,
            })
            self.assertEqual(response.status_code, 200)
            self.assertIn('"t": "o"', response.get_data(as_text=True))
            stream.assert_called_once_with(ANY, 400)


if __name__ == "__main__":
    unittest.main()