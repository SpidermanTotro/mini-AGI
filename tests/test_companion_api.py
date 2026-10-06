import json
import os
import tempfile
import unittest
from unittest.mock import patch

from companion.server import app


class CompanionAPITests(unittest.TestCase):
    def test_health_is_read_only(self):
        response = app.test_client().get("/api/v1/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json(), {"ok": True, "mode": "read-only"})

    def test_capabilities_keep_remote_control_disabled(self):
        response = app.test_client().get("/api/v1/capabilities")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["api_version"], 1)
        self.assertEqual(data["api_min_client_version"], 1)
        self.assertEqual(data["project"], "DragonForge")
        self.assertEqual(data["core"], "Greenlight Recur")
        self.assertEqual(data["platform"], "linux")
        self.assertTrue(data["features"]["status"])
        self.assertTrue(data["features"]["doctor"])
        self.assertFalse(data["features"]["training_control"])

    def test_status_reads_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "status.json")
            with open(path, "w") as f:
                json.dump({"training": True, "step": 42, "loss": 1.25,
                           "experts": 128, "vram_gb": 9.5,
                           "doctor": "healthy"}, f)
            with patch.dict(os.environ, {"GREENLIGHT_STATUS_JSON": path}):
                data = app.test_client().get("/api/v1/status").get_json()
        self.assertTrue(data["training"])
        self.assertEqual(data["step"], 42)
        self.assertEqual(data["experts"], 128)
        self.assertEqual(data["doctor"], "healthy")


if __name__ == "__main__":
    unittest.main()
