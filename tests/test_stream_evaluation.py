import unittest

import numpy as np
import torch

from minagi.stream import FileReader


class TinyEvalModel:
    def __init__(self, context):
        self.context = context
        self.forward_lengths = []

    def empty_caches(self):
        return []

    def __call__(self, x, y, caches=None, pos_offset=0):
        self.forward_lengths.append(x.shape[1])
        if pos_offset + x.shape[1] > self.context:
            raise ValueError("rotary table exceeded")
        return torch.zeros((1, x.shape[1], 1)), torch.tensor(1.0)


class FileReaderEvaluationTests(unittest.TestCase):
    def test_measure_caps_oversized_chunk_to_context(self):
        context = 16
        model = TinyEvalModel(context)
        data = np.arange(80, dtype=np.int64)
        reader = FileReader(model, data, "tiny", chunk=64, context=context, device="cpu")

        loss = reader.step(learn=False)

        self.assertIsNotNone(loss)
        self.assertEqual(reader.pos, context)
        self.assertEqual(model.forward_lengths, [context])
        self.assertLessEqual(reader.seen, context)


if __name__ == "__main__":
    unittest.main()
