import json
import os
import tempfile
import unittest

from minagi.history import append_jsonl, truncate_history


class TruncateHistoryTests(unittest.TestCase):
    def test_drops_rows_past_resume_point(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "history.jsonl")
            rows = [
                {"chars": 100, "step": 1},
                {"chars": 200, "step": 2},
                {"chars": 300, "step": 3},
            ]
            with open(path, "w") as f:
                for row in rows:
                    f.write(json.dumps(row) + "\n")

            self.assertEqual(truncate_history(path, 200), 1)
            with open(path) as f:
                kept = [json.loads(line) for line in f if line.strip()]
            self.assertEqual([row["chars"] for row in kept], [100, 200])

    def test_preserves_unparseable_rows(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "history.jsonl")
            with open(path, "w") as f:
                f.write('{"chars": 300}\n')
                f.write("legacy or damaged row\n")
                f.write('{"chars": 100}\n')

            self.assertEqual(truncate_history(path, 200), 1)
            with open(path) as f:
                content = f.read()
            self.assertIn("legacy or damaged row", content)
            self.assertIn('{"chars": 100}', content)
            self.assertNotIn('{"chars": 300}', content)

    def test_disabled_missing_and_zero_resume_are_noops(self):
        with tempfile.TemporaryDirectory() as td:
            missing = os.path.join(td, "missing.jsonl")
            self.assertEqual(truncate_history("", 100), 0)
            self.assertEqual(truncate_history(missing, 100), 0)

            path = os.path.join(td, "history.jsonl")
            with open(path, "w") as f:
                f.write('{"chars": 300}\n')
            self.assertEqual(truncate_history(path, 0), 0)
            with open(path) as f:
                self.assertEqual(f.read(), '{"chars": 300}\n')


class AppendJsonlTests(unittest.TestCase):
    def test_appends_rows_and_creates_parent_directory(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "nested", "history.jsonl")
            append_jsonl(path, {"step": 1, "chars": 100})
            append_jsonl(path, {"step": 2, "chars": 200})

            with open(path) as f:
                rows = [json.loads(line) for line in f]
            self.assertEqual(rows, [
                {"step": 1, "chars": 100},
                {"step": 2, "chars": 200},
            ])

    def test_compact_output_is_one_json_object_per_line(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "history.jsonl")
            append_jsonl(path, {"step": 1, "chars": 100}, compact=True)
            with open(path) as f:
                self.assertEqual(f.read(), '{"step":1,"chars":100}\\n')


if __name__ == "__main__":
    unittest.main()
