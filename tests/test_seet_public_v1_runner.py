"""Focused tests for the SEET-public-v1 reference runner."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tools"))

import run_seet_public_v1 as runner  # noqa: E402
from seet_public_v1_model import MyModel  # noqa: E402


class SplitTests(unittest.TestCase):
    def test_frozen_lists_are_exact_and_recording_disjoint(self) -> None:
        split = runner.load_and_validate_split()
        self.assertEqual(split.train_ids, runner.EXPECTED_TRAIN_IDS)
        self.assertEqual(split.validation_ids, runner.EXPECTED_VALIDATION_IDS)
        self.assertFalse(set(split.train_ids) & set(split.validation_ids))
        self.assertEqual(split.train_list_sha256, runner.EXPECTED_TRAIN_LIST_SHA256)
        self.assertEqual(
            split.validation_list_sha256,
            runner.EXPECTED_VALIDATION_LIST_SHA256,
        )

    def test_protocol_window_counts(self) -> None:
        self.assertEqual(len(runner.window_starts(runner.TRAIN_STRIDE)), 461)
        self.assertEqual(len(runner.window_starts(runner.VALIDATION_STRIDE)), 12)
        self.assertEqual(runner.window_starts(runner.VALIDATION_STRIDE)[-1], 440)


class DataTests(unittest.TestCase):
    def test_label_sampling_and_axis_normalisation(self) -> None:
        lines = []
        for label_index in range(500):
            lines.extend(["ignored", "ignored", "ignored", f"{label_index} {2 * label_index}"])
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "labels.txt"
            path.write_text("\n".join(lines) + "\n")
            labels = runner.parse_label_file(path)
        self.assertEqual(labels.shape, (500, 2))
        np.testing.assert_allclose(labels[10], (10 / 640, 20 / 480))

    def test_overlap_verification_detects_corruption(self) -> None:
        frames = np.arange(43, dtype=np.float32)[:, None, None]
        windows = np.stack([frames[i : i + 40] for i in range(4)], axis=0)
        vector = windows[:, :, None, :, :]
        self.assertEqual(runner.verify_train_overlap(vector, "all", 8, "ok"), 3)
        broken = vector.copy()
        broken[2, 5, 0, 0, 0] += 1
        with self.assertRaises(runner.BenchmarkInputError):
            runner.verify_train_overlap(broken, "all", 8, "broken")


class MetricTests(unittest.TestCase):
    def test_correct_and_legacy_axis_scaling_are_distinct(self) -> None:
        target = torch.zeros((1, 2, 2))
        prediction = torch.tensor([[[0.1, 0.0], [0.0, 0.1]]])
        correct = runner.pixel_distances(prediction, target)
        legacy = runner.pixel_distances(prediction, target, legacy_swapped=True)
        torch.testing.assert_close(correct, torch.tensor([[8.0, 6.0]]))
        torch.testing.assert_close(legacy, torch.tensor([[6.0, 8.0]]))

    def test_contiguous_smooth_l1_accepts_noncontiguous_prediction(self) -> None:
        base = torch.zeros((2, 40, 4), requires_grad=True)
        prediction = base[:, :, ::2]
        self.assertFalse(prediction.is_contiguous())
        loss = runner.smooth_l1(torch.nn.SmoothL1Loss(), prediction, torch.ones_like(prediction))
        loss.backward()
        self.assertIsNotNone(base.grad)

    def test_primary_threshold_is_strict(self) -> None:
        target = torch.zeros((1, 1, 2))
        prediction = torch.tensor([[[3.0 / 80.0, 0.0]]])
        accumulator = runner.MetricAccumulator(include_legacy=False)
        accumulator.update(torch.tensor(0.0), prediction, target)
        result = accumulator.result()
        self.assertEqual(result["p3_percent"], 0.0)
        self.assertEqual(result["p5_percent"], 100.0)

    def test_only_frozen_training_settings_are_protocol_compliant(self) -> None:
        config = {
            "epochs": 100,
            "batch_size": 16,
            "learning_rate": 0.001,
            "seed": 1,
            "deterministic": True,
            "overlap_check": "all",
        }
        self.assertTrue(runner.assess_protocol_compliance(config)["compliant"])
        config["epochs"] = 1
        compliance = runner.assess_protocol_compliance(config)
        self.assertFalse(compliance["compliant"])
        self.assertEqual(compliance["status"], "nonstandard_exploratory")

    @unittest.skipUnless(
        getattr(torch.backends, "mps", None) and torch.backends.mps.is_available(),
        "MPS is unavailable",
    )
    def test_mps_forward_and_loss_smoke(self) -> None:
        device = torch.device("mps")
        model = MyModel().to(device).train()
        frames = torch.zeros((1, 1, 1, 60, 80), device=device)
        targets = torch.zeros((1, 1, 2), device=device)
        prediction = model(frames)
        loss = runner.smooth_l1(torch.nn.SmoothL1Loss(), prediction, targets)
        loss.backward()
        self.assertTrue(torch.isfinite(loss).item())


if __name__ == "__main__":
    unittest.main()
