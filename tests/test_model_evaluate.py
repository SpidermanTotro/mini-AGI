import unittest

import torch

from minagi.evaluate import evaluate_cases, exact_numeric_answer
from minagi.tokenizer import ByteTokenizer


class FakeModel:
    def generate(self, tokens, max_new_tokens):
        assert not torch.is_grad_enabled()
        marker = torch.tensor([[ord("!")]], dtype=torch.long, device=tokens.device)
        return torch.cat((tokens, marker), dim=1)


class ModelEvaluationTests(unittest.TestCase):
    def test_exact_numeric_scoring_checks_only_final_answer_line(self):
        self.assertEqual(exact_numeric_answer("<think>work</think>\n1056\n"), "1056")
        self.assertEqual(exact_numeric_answer("95.\n"), "95")
        self.assertIsNone(exact_numeric_answer("The answer is 95."))
        self.assertIsNone(exact_numeric_answer("\n"))

    def test_probe_records_only_generated_completion(self):
        cases = [{"name": "tiny", "prompt": "prompt"}]
        result = evaluate_cases(FakeModel(), "cpu", max_new_tokens=4, cases=cases)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["name"], "tiny")
        self.assertEqual(result[0]["prompt"], "prompt")
        self.assertEqual(result[0]["completion"], "!")
        self.assertEqual(ByteTokenizer().decode(
            ByteTokenizer().encode("!").ids), "!")


if __name__ == "__main__":
    unittest.main()
