import math
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from src.experiments.pretrain.protocol import crystal_system, grid, nested_ids, run_name
from src.experiments.pretrain.data import official_voigt, standard_voigt
from src.experiments.pretrain.train import benchmark_inference, metrics
from src.experiments.pretrain.features import extract_embedding, geometry_input


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

    def test_fresh_inference_requires_prediction_reproduction(self):
        args = SimpleNamespace(smoke=True, device="cpu")
        dataset = SimpleNamespace(split_manifest=SimpleNamespace(test=("a",)), by_id=lambda sid: sid)
        prediction = torch.eye(3).unsqueeze(0)
        with patch("src.experiments.pretrain.train.dataset_from_records", return_value=dataset), \
             patch("src.experiments.pretrain.train.prepare_graph", return_value={}), \
             patch("src.experiments.pretrain.train._predict", return_value=prediction):
            result = benchmark_inference(args,"dielectric",{},[],None,None,None,None,None,False,
                                         {"a":prediction[0]})
            self.assertEqual(result["samples"][0]["reproduction_max_abs"], 0)
            self.assertGreaterEqual(result["mean_seconds_per_structure"], 0)
            with self.assertRaisesRegex(ValueError, "reproduce"):
                benchmark_inference(args,"dielectric",{},[],None,None,None,None,None,False,
                                    {"a":prediction[0]+1})

    def test_native_geometry_matches_legacy_parity_formula(self):
        from src.backbones.parity import SO3FeatureBatch, SO3Layout, SO3Term, InversionPairedReynolds
        from src.baselines.gmtnet.runner import dpa4_invariant_node_embedding
        from src.graphs import build_periodic_graph
        from src.symmetry import canonicalize_structure
        layout=SO3Layout((SO3Term(1,0,"s"),SO3Term(1,1,"v")))
        class Extractor(torch.nn.Module):
            def forward(self,graph):
                x=graph.positions
                return SO3FeatureBatch(torch.cat((x.square().sum(1,keepdim=True),x),1),layout,graph.node_batch)
        extractor=Extractor()
        adapter=SimpleNamespace(extractor=extractor,parity=InversionPairedReynolds(extractor,layout),
                                resource=SimpleNamespace(cutoff_angstrom=1.0))
        sample=SimpleNamespace(cartesian_positions=torch.tensor([[.13,.27,.38],[.61,.74,.82]]),
                               lattice=torch.eye(3)*3,atomic_numbers=torch.tensor([6,8]))
        canonical=canonicalize_structure(sample.cartesian_positions,sample.lattice,sample.atomic_numbers)
        graph=build_periodic_graph(canonical.canonical_positions,canonical.canonical_cell,sample.atomic_numbers,1.)
        reference=adapter.parity(graph)
        expected=dpa4_invariant_node_embedding(reference.node_features,reference.node_layout)
        actual=extract_embedding(adapter,sample,device="cpu")
        torch.testing.assert_close(actual,expected,atol=0,rtol=0)
        carrier=geometry_input(sample.cartesian_positions,sample.lattice,sample.atomic_numbers,1.)
        self.assertEqual(carrier.num_edges,0)
        self.assertGreater(graph.num_edges,0)


if __name__ == "__main__":
    unittest.main()
