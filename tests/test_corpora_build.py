import unittest
from unittest.mock import patch

import corpora.build as build


class CorporaBuildTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
