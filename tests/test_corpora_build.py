import unittest
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import corpora.build as build


class CorporaBuildTests(unittest.TestCase):
    @patch("corpora.build.os.chdir")
    @patch("corpora.build._has_files", return_value=True)
    def test_main_fails_when_forced_builder_fails_with_existing_files(
            self, has_files, chdir):
        from unittest.mock import Mock

        builder = Mock(return_value=1)
        with patch.dict(build.BUILDERS, {"code": (builder, "stale-code")}), \
                patch.object(sys, "argv", ["corpora", "--only", "code",
                                            "--force"]):
            result = build.main()

        self.assertEqual(result, 1)
        builder.assert_called_once_with(0)

    def test_chat_fallback_moves_shards_into_the_expected_lane(self):
        def fake_sub(*args):
            if args[0] == "fetch":
                return 1
            if args[0] == "chat":
                return 0
            if args[0] == "expand":
                source = Path("data/train/chat")
                source.mkdir(parents=True)
                (source / "chat-00000.txt").write_text(
                    "fallback shard", encoding="utf-8")
                return 0
            raise AssertionError(f"unexpected target: {args[0]}")

        with tempfile.TemporaryDirectory() as root:
            previous = Path.cwd()
            try:
                os.chdir(root)
                with patch("corpora.build._sub", side_effect=fake_sub):
                    self.assertEqual(build.build_chat(128), 0)
                self.assertFalse(Path("data/train/chat/chat-00000.txt").exists())
                self.assertEqual(
                    Path("data/train/chat/hermes/chat-00000.txt").read_text(
                        encoding="utf-8"),
                    "fallback shard")
            finally:
                os.chdir(previous)

    @patch("corpora.build._sub")
    def test_build_chat_falls_back_to_local_generator_when_fetch_fails(self, sub):
        def fake_sub(*args):
            target = args[0]
            if target == "fetch":
                return 1
            if target == "chat":
                return 0
            if target == "expand":
                return 0
            raise AssertionError(f"unexpected target: {target}")

        sub.side_effect = fake_sub

        rc = build.build_chat(128)

        self.assertEqual(rc, 0)
        self.assertEqual(sub.call_count, 3)
        self.assertEqual(sub.call_args_list[0].args[0], "fetch")
        self.assertEqual(sub.call_args_list[1].args[0], "chat")
        self.assertEqual(sub.call_args_list[2].args[0], "expand")


    @patch("corpora.build._generated")
    def test_self_knowledge_has_dedicated_lane(self, generated):
        generated.return_value = 0

        self.assertEqual(build.build_self_knowledge(1234), 0)

        generated.assert_called_once_with(
            "chat", "self-knowledge", "--out", "data_self_chat_char",
            "--conversations", 1234, "--val", 200)

        from corpora import expand
        mapping = {name: src for name, src, _ in expand.SOURCES}
        self.assertEqual(mapping["self-knowledge"], "data_self_chat_char")
        self.assertNotEqual(mapping["self-knowledge"], mapping["chat"])

if __name__ == "__main__":
    unittest.main()
