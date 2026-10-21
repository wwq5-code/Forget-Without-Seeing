"""Deterministic unit tests; no model downloads, training datasets or GPU needed.

python -m unittest On_CIFAR100.test_functional_research_audit -v
"""
import inspect
import math
import unittest
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

from On_CIFAR100 import functional_research_audit as audit
from On_CIFAR100 import unlearning_clip_by_subliminal as core


class ResearchAuditTests(unittest.TestCase):
    def test_auc_and_ties(self):
        y = np.array([0, 0, 1, 1])
        self.assertEqual(audit.roc_auc_score(y, np.ones(4)), .5)
        self.assertEqual(audit.roc_auc_score(y, np.array([0, 1, 2, 3])), 1.)
        self.assertEqual(audit.roc_auc_score(y, np.array([3, 2, 1, 0])), 0.)

    def test_attack_partitions_and_no_test_direction_selection(self):
        labels = np.r_[np.zeros(40, dtype=int), np.ones(40, dtype=int)]
        split = audit.split_attack_indices(labels, 7)
        self.assertEqual(len(set().union(*(set(s) for s in split))), len(labels))
        self.assertEqual(sum(map(len, split)), len(labels))
        features = labels[:, None].astype(float)
        features[split[2]] = 1 - features[split[2]]
        result, _ = audit.fit_attack(features, labels, split)
        self.assertEqual(result["auc"], 0.)  # Must not flip to 1 using test labels.
        self.assertFalse(result["low_fpr_resolution_sufficient"])

    def test_projection_clipping_and_add_remove_sensitivity(self):
        torch.manual_seed(42)
        raw = torch.randn(8, 4, 6) * 10
        q = torch.linalg.qr(torch.randn(6, 3), mode="reduced")[0]
        args = SimpleNamespace(dp_clip_norm=.2, dp_delta=1e-5)
        decoded, released, metadata = audit.private_message(raw, 8, "full_raw", q, math.inf, args)
        self.assertTrue(torch.allclose(decoded, raw.mean(0)))
        self.assertFalse(metadata["projection_applied"])
        decoded_p, released_p, _ = audit.private_message(raw, 8, "projected_raw", q, math.inf, args)
        self.assertTrue(torch.allclose(decoded_p, raw.mean(0) @ q @ q.t(), atol=2e-6))
        _, clipped, meta = audit.private_message(raw, 8, "clipped", q, math.inf, args)
        _, removed, _ = audit.private_message(raw[:-1], 8, "clipped", q, math.inf, args)
        self.assertLessEqual(float((clipped-removed).norm()), args.dp_clip_norm/8 + 1e-6)
        self.assertEqual(meta["clipped_fraction_audit_only"], 1.)
        a, _, meta_a = audit.private_message(raw, 8, "noisy", q, 2., args)
        b, _, _ = audit.private_message(raw, 8, "noisy", q, 2., args)
        self.assertFalse(torch.equal(a, b))
        self.assertTrue(meta_a["formal_dp"])
        self.assertAlmostEqual(meta_a["achieved_epsilon"], 2., places=6)

    def test_hard_preservation_and_receiver_boundary(self):
        torch.manual_seed(0)
        args = core.build_parser().parse_args([])
        args.student_steps = 2
        args.personal_preservation_budget = 1e-4
        args.preservation_mode = "hard"
        base = torch.eye(8)
        basis = torch.linalg.qr(torch.randn(8, 3), mode="reduced")[0]
        features = torch.randn(12, 8)
        text = F.normalize(torch.randn(4, 8), dim=-1)
        receiver = audit.ReceiverState(base, basis, torch.zeros(3, 8), features,
                                       (torch.randn(16, 8), torch.randn(16, 8), torch.zeros(16, dtype=torch.long)),
                                       text, torch.tensor(3.))
        student, details = receiver.reconstruct(torch.randn(12, 8)*10, args, 43)
        self.assertLessEqual(details["preservation_violation"], 2e-7)
        self.assertLessEqual(details["personal_preservation_loss"], args.personal_preservation_budget+2e-7)
        self.assertTrue(torch.isfinite(student).all())
        self.assertEqual(list(inspect.signature(receiver.reconstruct).parameters), ["message", "args", "seed"])

    def test_model_seed_aggregation_not_noise_pseudoreplication(self):
        rows = [{"setting_id": "x", "seed": seed, "configuration": {}, "metrics": {"running_time_seconds": value}}
                for seed, values in ((43, (1, 3)), (44, (5, 5))) for value in values]
        metric = audit.aggregate_metrics(rows)[0]["metrics"]["running_time_seconds"]
        self.assertEqual(metric["n_model_seeds"], 2)
        self.assertEqual(metric["n_observations"], 4)
        self.assertEqual(metric["mean"], 3.5)
        self.assertAlmostEqual(metric["sample_std"], 3/math.sqrt(2))


if __name__ == "__main__":
    unittest.main()
