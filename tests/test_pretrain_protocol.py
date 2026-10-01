import math
import unittest

import torch

from src.experiments.pretrain.protocol import crystal_system, grid, nested_ids, run_name
from src.experiments.pretrain.data import official_voigt, standard_voigt
from src.experiments.pretrain.train import metrics


class ProtocolTests(unittest.TestCase):
    def test_nested_paired_fixed_splits(self):
        rows = [{"record_id": f"{split}{i}", "split": split}
                for split, n in (("train", 103), ("validation", 13), ("test", 17)) for i in range(n)]
        splits = [nested_ids(rows, f) for f in (25, 50, 75, 100)]
        self.assertEqual([len(s["train"]) for s in splits], [25, 51, 77, 103])
        for low, high in zip(splits, splits[1:]):
            self.assertLess(set(low["train"]), set(high["train"]))
            self.assertEqual(low["test"], high["test"])
            self.assertEqual(low["validation"], high["validation"])
        for spec in grid():
            self.assertEqual(nested_ids(rows, spec["fraction"]), splits[(spec["fraction"]//25)-1])
        self.assertNotEqual(nested_ids(rows, 25, sampling_seed=1)["train"], splits[0]["train"])

    def test_reject_overlap_and_bad_fraction(self):
        rows = [{"record_id": "x", "split": s} for s in ("train", "validation", "test")]
        with self.assertRaises(ValueError):
            nested_ids(rows, 25)
        with self.assertRaises(ValueError):
            nested_ids(rows, 35)

    def test_grid_unique(self):
        self.assertEqual(len(grid()), 48)
        self.assertEqual(len({run_name(s) for s in grid()}), 48)
        for a, b in zip(grid()[::2], grid()[1::2]):
            self.assertEqual({k:v for k,v in a.items() if k != "model"},
                             {k:v for k,v in b.items() if k != "model"})

    def test_crystal_system_boundaries(self):
        self.assertEqual(crystal_system(142), "Tetragonal")
        self.assertEqual(crystal_system(143), "Trigonal")
        self.assertEqual(crystal_system(194), "Hexagonal")
        self.assertEqual(crystal_system(195), "Cubic")
        with self.assertRaises(ValueError):
            crystal_system(0)

    def test_elastic_component_convention(self):
        tensor = torch.arange(81).reshape(3,3,3,3)
        official = official_voigt(tensor)
        standard = standard_voigt(official)
        self.assertEqual(official[3,4], tensor[0,1,1,2])
        self.assertEqual(standard[3,5], tensor[1,2,0,1])
        self.assertEqual(standard[4,4], tensor[0,2,0,2])

    def test_metrics_hand_computed(self):
        target = torch.eye(3).repeat(2,1,1)*10
        prediction = target.clone()
        prediction[0,0,0] += 3
        got = metrics(prediction, target, "dielectric")
        self.assertAlmostEqual(got["fnorm"], 1.5)
        self.assertAlmostEqual(got["rmse"], math.sqrt(9/18))
        self.assertEqual(got["ewt_25"], 100)
        self.assertEqual(got["ewt_10"], 50)
        self.assertAlmostEqual(got["mae"], 3/18)


if __name__ == "__main__":
    unittest.main()
