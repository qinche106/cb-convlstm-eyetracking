#!/usr/bin/env python3
"""Reproducible vanilla-ConvLSTM runner for the frozen SEET-public-v1 split.

This runner uses the repository's existing ``MyModel`` architecture.  It does
not implement or claim results for change-based (CB) ConvLSTM execution.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import random
import re
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import cv2
import numpy as np
import tables
import torch
from torch import Tensor, nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset

from seet_public_v1_model import MyModel, parameter_count


SPLIT_ID = "SEET-public-v1"
DATA_ARTIFACT_ID = "SEET-eventframes-v0"
PROTOCOL_ID = "SEET-public-v1-convlstm-code-v1"
MODEL_ID = "vanilla_convlstm_mymodel"
MODEL_SCOPE = "vanilla ConvLSTM only; not a CB-ConvLSTM implementation"
METRIC_ID = "euclidean_xy_80x60_px_strict_lt_v1"
LEGACY_METRIC_ID = "legacy_swapped_xy_60x80_px_v0"
SEQUENCE_LENGTH, TRAIN_STRIDE, VALIDATION_STRIDE, CHUNK_SIZE = 40, 1, 40, 500
INPUT_HEIGHT, INPUT_WIDTH, RAW_HEIGHT, RAW_WIDTH = 60, 80, 180, 240
OFFICIAL_EPOCHS, OFFICIAL_BATCH_SIZE, OFFICIAL_LEARNING_RATE, OFFICIAL_SEED = 100, 16, 0.001, 1

EXPECTED_TRAIN_IDS = (
    "subj_18_vid_2", "subj_10_vid_8", "subj_12_vid_1", "subj_12_vid_2",
    "subj_12_vid_9", "subj_13_vid_1", "subj_13_vid_2", "subj_14_vid_10",
    "subj_15_vid_2", "subj_21_vid_11", "subj_1_vid_1", "subj_1_vid_9",
    "subj_2_vid_4", "subj_6_vid_5", "subj_7_vid_18", "subj_8_vid_2",
)
EXPECTED_VALIDATION_IDS = ("subj_16_vid_1", "subj_18_vid_7")
EXPECTED_TRAIN_LIST_SHA256 = "d0a3e202fcd084f55bcd265aca0abfeadb33d90b8989af08fc47a27eb6aeaf50"
EXPECTED_VALIDATION_LIST_SHA256 = "32a21bfb5105f89568ff266e3a0e9d6aa54fb5d5aa1b56775360ac76de4d400e"

RECORDING_RE = re.compile(r"^subj_[0-9]+_vid_[0-9]+$")
REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = Path(__file__).resolve()
MODEL_PATH = RUNNER_PATH.with_name("seet_public_v1_model.py")
TRAIN_LIST_PATH = REPO_ROOT / "splits" / SPLIT_ID / "train_files.txt"
VALIDATION_LIST_PATH = REPO_ROOT / "splits" / SPLIT_ID / "val_files.txt"


class BenchmarkInputError(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(value: np.ndarray) -> str:
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(json.dumps(list(value.shape)).encode("ascii"))
    digest.update(value.tobytes())
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def load_recording_list(path: Path) -> Tuple[str, ...]:
    values: List[str] = []
    for line_number, raw in enumerate(path.read_text().splitlines(), 1):
        value = raw.strip()
        if not value:
            continue
        if RECORDING_RE.fullmatch(value) is None:
            raise BenchmarkInputError(f"{path}:{line_number}: invalid ID {value!r}")
        if value in values:
            raise BenchmarkInputError(f"{path}:{line_number}: duplicate ID {value!r}")
        values.append(value)
    return tuple(values)


@dataclass(frozen=True)
class SplitDefinition:
    train_ids: Tuple[str, ...]
    validation_ids: Tuple[str, ...]
    train_list_sha256: str
    validation_list_sha256: str

    def as_dict(self) -> Dict[str, Any]:
        return {
            "split_id": SPLIT_ID,
            "train_ids": list(self.train_ids),
            "validation_ids": list(self.validation_ids),
            "train_recordings": len(self.train_ids),
            "validation_recordings": len(self.validation_ids),
            "recording_overlap": sorted(set(self.train_ids) & set(self.validation_ids)),
            "train_list_sha256": self.train_list_sha256,
            "validation_list_sha256": self.validation_list_sha256,
        }


def load_and_validate_split() -> SplitDefinition:
    train = load_recording_list(TRAIN_LIST_PATH)
    val = load_recording_list(VALIDATION_LIST_PATH)
    train_hash, val_hash = sha256_path(TRAIN_LIST_PATH), sha256_path(VALIDATION_LIST_PATH)
    if train != EXPECTED_TRAIN_IDS or train_hash != EXPECTED_TRAIN_LIST_SHA256:
        raise BenchmarkInputError("canonical train list differs from frozen SEET-public-v1")
    if val != EXPECTED_VALIDATION_IDS or val_hash != EXPECTED_VALIDATION_LIST_SHA256:
        raise BenchmarkInputError("canonical validation list differs from frozen SEET-public-v1")
    if set(train) & set(val):
        raise BenchmarkInputError("recording-level train/validation leakage detected")
    return SplitDefinition(train, val, train_hash, val_hash)


def window_starts(stride: int) -> np.ndarray:
    return np.arange(0, CHUNK_SIZE - SEQUENCE_LENGTH + 1, stride, dtype=np.int64)


def normalise_resized_frame(frame: np.ndarray) -> np.ndarray:
    resized = cv2.resize(frame, (INPUT_WIDTH, INPUT_HEIGHT))
    mean, standard_deviation = np.mean(resized), np.std(resized)
    if standard_deviation != 0:
        resized = (resized - mean) / (standard_deviation + 1e-10)
    return np.asarray(resized, dtype=np.float32)


def parse_label_file(path: Path) -> np.ndarray:
    raw_lines = path.read_text().splitlines()
    selected = raw_lines[3::4]
    if len(selected) != CHUNK_SIZE:
        raise BenchmarkInputError(
            f"{path}: expected 500 labels after lines[3::4], found {len(selected)}"
        )
    labels = np.empty((CHUNK_SIZE, 2), dtype=np.float32)
    for index, raw in enumerate(selected):
        fields = raw.split()
        if len(fields) != 2:
            raise BenchmarkInputError(f"{path}:{4 + index * 4}: expected x y")
        try:
            labels[index] = float(fields[0]), float(fields[1])
        except ValueError as exc:
            raise BenchmarkInputError(f"{path}:{4 + index * 4}: non-numeric label") from exc
    if not np.isfinite(labels).all():
        raise BenchmarkInputError(f"{path}: non-finite labels")
    labels[:, 0] /= 640.0
    labels[:, 1] /= 480.0
    return labels


def make_label_windows(labels: np.ndarray, stride: int) -> np.ndarray:
    starts = window_starts(stride)
    indices = starts[:, None] + np.arange(SEQUENCE_LENGTH)[None, :]
    return np.ascontiguousarray(labels[indices], dtype=np.float32)


def sampled_boundaries(number_of_windows: int, count: int) -> List[int]:
    if count <= 0:
        return []
    return sorted(set(int(x) for x in np.linspace(1, number_of_windows - 1, min(count, number_of_windows - 1), dtype=int)))


def verify_train_overlap(vector: Any, mode: str, samples: int, recording_id: str) -> int:
    if mode == "none":
        return 0
    if mode == "all":
        previous = np.asarray(vector[0, :, 0, :, :])
        checked = 0
        for current_index in range(1, int(vector.shape[0])):
            current = np.asarray(vector[current_index, :, 0, :, :])
            if not np.array_equal(previous[1:], current[:-1]):
                raise BenchmarkInputError(
                    f"{recording_id}: windows {current_index - 1}/{current_index} "
                    "do not share an identical 39-frame overlap"
                )
            previous = current
            checked += 1
        return checked
    boundaries: Iterable[int]
    if mode == "sample":
        boundaries = sampled_boundaries(int(vector.shape[0]), samples)
    else:
        raise ValueError(mode)
    checked = 0
    for current_index in boundaries:
        previous = np.asarray(vector[current_index - 1, 1:, 0, :, :])
        current = np.asarray(vector[current_index, :-1, 0, :, :])
        if not np.array_equal(previous, current):
            raise BenchmarkInputError(
                f"{recording_id}: windows {current_index - 1}/{current_index} "
                "do not share an identical 39-frame overlap"
            )
        checked += 1
    return checked


def reconstruct_frame_cache(
    path: Path, partition: str, recording_id: str, overlap_check: str, overlap_samples: int
) -> Tuple[np.ndarray, Dict[str, Any]]:
    stride = TRAIN_STRIDE if partition == "train" else VALIDATION_STRIDE
    expected_shape = (len(window_starts(stride)), SEQUENCE_LENGTH, 1, RAW_HEIGHT, RAW_WIDTH)
    with tables.open_file(str(path), "r") as handle:
        if not hasattr(handle.root, "vector"):
            raise BenchmarkInputError(f"{path}: missing /vector")
        vector = handle.root.vector
        shape = tuple(int(x) for x in vector.shape)
        if shape != expected_shape:
            raise BenchmarkInputError(f"{path}:/vector shape {shape}, expected {expected_shape}")
        frame_count = CHUNK_SIZE if partition == "train" else shape[0] * SEQUENCE_LENGTH
        cache = np.empty((frame_count, 1, INPUT_HEIGHT, INPUT_WIDTH), np.float32)
        if partition == "train":
            for frame_index in range(SEQUENCE_LENGTH):
                cache[frame_index, 0] = normalise_resized_frame(vector[0, frame_index, 0])
            for window_index in range(1, shape[0]):
                cache[SEQUENCE_LENGTH - 1 + window_index, 0] = normalise_resized_frame(
                    vector[window_index, -1, 0]
                )
            overlap_count = verify_train_overlap(vector, overlap_check, overlap_samples, recording_id)
        else:
            output_index = 0
            for window_index in range(shape[0]):
                for frame_index in range(SEQUENCE_LENGTH):
                    cache[output_index, 0] = normalise_resized_frame(vector[window_index, frame_index, 0])
                    output_index += 1
            overlap_count = 0
        metadata = {
            "recording_id": recording_id,
            "partition": partition,
            "relative_h5": f"{partition}/{recording_id}.h5",
            "h5_size_bytes": path.stat().st_size,
            "h5_sha256": sha256_path(path),
            "h5_vector_shape": list(shape),
            "h5_vector_dtype": str(vector.dtype),
            "h5_chunkshape": [int(x) for x in vector.chunkshape],
            "reconstructed_frames": frame_count,
            "overlap_check": overlap_check if partition == "train" else "not_applicable",
            "overlap_boundaries_checked": overlap_count,
            "effective_frames_sha256": sha256_array(cache),
        }
    return cache, metadata


@dataclass
class RecordingCache:
    recording_id: str
    frames: np.ndarray
    labels: np.ndarray
    metadata: Dict[str, Any]


class SEETWindowDataset(Dataset):
    def __init__(
        self, recording_ids: Sequence[str], partition: str, data_root: Path,
        label_root: Path, overlap_check: str, overlap_samples: int,
    ) -> None:
        self.partition = partition
        self.stride = TRAIN_STRIDE if partition == "train" else VALIDATION_STRIDE
        self.starts = window_starts(self.stride)
        self.caches: List[RecordingCache] = []
        ordered_recording_ids = tuple(sorted(recording_ids))
        for number, recording_id in enumerate(ordered_recording_ids, 1):
            print(f"[{partition} cache {number}/{len(ordered_recording_ids)}] {recording_id}", flush=True)
            h5_path = data_root / partition / f"{recording_id}.h5"
            label_path = label_root / f"{recording_id}.txt"
            frames, metadata = reconstruct_frame_cache(
                h5_path, partition, recording_id, overlap_check, overlap_samples
            )
            labels = make_label_windows(parse_label_file(label_path), self.stride)
            metadata.update({
                "label_file_sha256": sha256_path(label_path),
                "effective_labels_sha256": sha256_array(labels),
                "sequence_windows": len(self.starts),
            })
            metadata["effective_recording_sha256"] = sha256_json({
                "id": recording_id,
                "raw_h5": metadata["h5_sha256"],
                "raw_label": metadata["label_file_sha256"],
                "frames": metadata["effective_frames_sha256"],
                "labels": metadata["effective_labels_sha256"],
            })
            self.caches.append(RecordingCache(recording_id, frames, labels, metadata))
        self.fingerprint_sha256 = sha256_json(
            [cache.metadata["effective_recording_sha256"] for cache in self.caches]
        )

    def __len__(self) -> int:
        return len(self.caches) * len(self.starts)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        recording_index, sample_index = divmod(index, len(self.starts))
        cache = self.caches[recording_index]
        start = int(self.starts[sample_index])
        frames = cache.frames[start:start + SEQUENCE_LENGTH]
        if len(frames) != SEQUENCE_LENGTH:
            raise AssertionError("incomplete cached sequence")
        return torch.from_numpy(frames), torch.from_numpy(cache.labels[sample_index])

    def metadata(self) -> Dict[str, Any]:
        return {
            "partition": self.partition,
            "dataset_order": "lexicographic_recording_id_then_window_start",
            "recordings": len(self.caches),
            "sequences": len(self),
            "stride": self.stride,
            "effective_partition_sha256": self.fingerprint_sha256,
            "recording_inputs": [cache.metadata for cache in self.caches],
        }


def validate_input_inventory(
    data_root: Path, label_root: Path, split: SplitDefinition
) -> Dict[str, Any]:
    if not data_root.is_dir() or not label_root.is_dir():
        raise BenchmarkInputError("data-root and label-root must be directories")
    inventory: Dict[str, Any] = {}
    for partition, expected in (("train", split.train_ids), ("val", split.validation_ids)):
        directory = data_root / partition
        if not directory.is_dir():
            raise BenchmarkInputError(f"missing partition directory {directory}")
        actual = {path.stem for path in directory.glob("*.h5") if path.is_file()}
        missing = set(expected) - actual
        if missing:
            raise BenchmarkInputError(
                f"{directory}: missing official H5 files: {', '.join(sorted(missing))}"
            )
        inventory[partition] = {
            "official_ids": list(expected),
            "extra_h5_ids_ignored": sorted(actual - set(expected)),
        }
    all_ids = split.train_ids + split.validation_ids
    missing_labels = [x for x in all_ids if not (label_root / f"{x}.txt").is_file()]
    if missing_labels:
        raise BenchmarkInputError(f"missing labels: {', '.join(missing_labels)}")
    actual_labels = {path.stem for path in label_root.glob("*.txt") if path.is_file()}
    inventory["labels"] = {
        "official_ids": list(all_ids),
        "extra_txt_ids_ignored": sorted(actual_labels - set(all_ids)),
    }
    return inventory


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise BenchmarkInputError("CUDA requested but unavailable")
    if requested == "mps" and not (
        getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
    ):
        raise BenchmarkInputError("MPS requested but unavailable")
    return torch.device(requested)


def seed_everything(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def seed_worker(_: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_loader(
    dataset: SEETWindowDataset, batch_size: int, shuffle: bool, seed: int,
    num_workers: int, device: torch.device,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
        pin_memory=device.type == "cuda", generator=generator,
        worker_init_fn=seed_worker if num_workers else None,
        persistent_workers=num_workers > 0,
    )


def pixel_distances(
    predictions: Tensor, targets: Tensor, legacy_swapped: bool = False
) -> Tensor:
    scale = predictions.new_tensor((60.0, 80.0) if legacy_swapped else (80.0, 60.0))
    return torch.linalg.vector_norm((predictions - targets) * scale, dim=-1)


class MetricAccumulator:
    def __init__(self, include_legacy: bool) -> None:
        self.include_legacy = include_legacy
        self.loss_sum, self.loss_values = 0.0, 0
        self.correct_sum, self.correct_count = 0.0, 0
        self.correct_within = {3: 0, 5: 0, 10: 0}
        self.legacy_sum, self.legacy_count = 0.0, 0
        self.legacy_within = {3: 0, 5: 0, 10: 0}

    def update(self, loss: Tensor, predictions: Tensor, targets: Tensor) -> None:
        self.loss_sum += float(loss.item()) * targets.numel()
        self.loss_values += targets.numel()
        correct = pixel_distances(predictions.detach(), targets.detach())
        self.correct_sum += float(correct.sum().item())
        self.correct_count += correct.numel()
        for threshold in self.correct_within:
            self.correct_within[threshold] += int((correct < threshold).sum().item())
        if self.include_legacy:
            legacy = pixel_distances(predictions.detach(), targets.detach(), True)
            self.legacy_sum += float(legacy.sum().item())
            self.legacy_count += legacy.numel()
            for threshold in self.legacy_within:
                self.legacy_within[threshold] += int((legacy <= threshold).sum().item())

    def result(self) -> Dict[str, Any]:
        if not self.correct_count:
            raise RuntimeError("empty evaluation loader")
        result: Dict[str, Any] = {
            "loss": self.loss_sum / self.loss_values,
            "metric_id": METRIC_ID,
            "mean_error_px": self.correct_sum / self.correct_count,
            "evaluated_points": self.correct_count,
        }
        for threshold, count in self.correct_within.items():
            result[f"p{threshold}_percent"] = 100.0 * count / self.correct_count
        if self.include_legacy:
            legacy: Dict[str, Any] = {
                "metric_id": LEGACY_METRIC_ID,
                "mean_error_px": self.legacy_sum / self.legacy_count,
                "evaluated_points": self.legacy_count,
            }
            for threshold, count in self.legacy_within.items():
                legacy[f"p{threshold}_percent"] = 100.0 * count / self.legacy_count
            result["legacy_swapped_metric"] = legacy
        return result


def smooth_l1(criterion: nn.Module, predictions: Tensor, targets: Tensor) -> Tensor:
    """Contiguous inputs avoid a known MPS SmoothL1 striding failure."""
    return criterion(predictions.contiguous(), targets.contiguous())


def evaluate(
    model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device,
    include_legacy: bool,
) -> Dict[str, Any]:
    model.eval()
    metrics = MetricAccumulator(include_legacy)
    with torch.no_grad():
        for frames, targets in loader:
            frames = frames.to(device, torch.float32, non_blocking=True)
            targets = targets.to(device, torch.float32, non_blocking=True)
            predictions = model(frames)
            loss = smooth_l1(criterion, predictions, targets)
            metrics.update(loss, predictions, targets)
    return metrics.result()


def train_one_epoch(
    model: nn.Module, loader: DataLoader, criterion: nn.Module,
    optimizer: torch.optim.Optimizer, device: torch.device,
) -> float:
    # Important: validation leaves the model in eval mode, so reset every epoch.
    model.train()
    total, values = 0.0, 0
    for batch_index, (frames, targets) in enumerate(loader, 1):
        frames = frames.to(device, torch.float32, non_blocking=True)
        targets = targets.to(device, torch.float32, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        predictions = model(frames)
        loss = smooth_l1(criterion, predictions, targets)
        loss.backward()
        optimizer.step()
        total += float(loss.item()) * targets.numel()
        values += targets.numel()
        if batch_index == 1 or batch_index % 25 == 0 or batch_index == len(loader):
            print(f"  batch {batch_index}/{len(loader)} loss={loss.item():.6f}", flush=True)
    return total / values


def git_source_metadata() -> Dict[str, Any]:
    def git(*args: str) -> Optional[str]:
        try:
            return subprocess.run(
                ["git", *args], cwd=REPO_ROOT, check=True, capture_output=True, text=True
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    commit, status = git("rev-parse", "HEAD"), git("status", "--porcelain=v1")
    files = {
        str(RUNNER_PATH.relative_to(REPO_ROOT)): sha256_path(RUNNER_PATH),
        str(MODEL_PATH.relative_to(REPO_ROOT)): sha256_path(MODEL_PATH),
    }
    return {
        "source_commit": commit,
        "source_dirty": bool(status) if status is not None else None,
        "source_status_porcelain": status.splitlines() if status else [],
        "source_file_sha256": files,
        "source_tree_fingerprint_sha256": sha256_json({"commit": commit, "files": files}),
    }


def protocol_metadata() -> Dict[str, Any]:
    return {
        "protocol_id": PROTOCOL_ID,
        "data_artifact_id": DATA_ARTIFACT_ID,
        "sequence_length": SEQUENCE_LENGTH,
        "train_stride": TRAIN_STRIDE,
        "validation_stride": VALIDATION_STRIDE,
        "chunk_size": CHUNK_SIZE,
        "input_height": INPUT_HEIGHT,
        "input_width": INPUT_WIDTH,
        "label_sampling": "lines[3::4]",
        "label_normalisation": {"x_divisor": 640, "y_divisor": 480},
        "frame_preprocessing": "cv2.resize(80x60), then per-frame standardisation",
        "optimizer": "Adam",
        "official_training_config": {
            "epochs": OFFICIAL_EPOCHS,
            "batch_size": OFFICIAL_BATCH_SIZE,
            "learning_rate": OFFICIAL_LEARNING_RATE,
            "seed": OFFICIAL_SEED,
            "adam_betas": [0.9, 0.999],
            "adam_eps": 1e-8,
            "weight_decay": 0.0,
            "scheduler": None,
            "deterministic_algorithms": True,
            "overlap_check": "all",
        },
        "loss_function": "SmoothL1Loss(beta=1, reduction=mean)",
        "checkpoint_rule": "minimum validation SmoothL1Loss; first epoch wins exact ties",
        "correct_metric": {
            "metric_id": METRIC_ID,
            "definition": "Euclidean distance after x*80,y*60; pN is percent strictly < N px",
        },
        "legacy_metric": {
            "metric_id": LEGACY_METRIC_ID,
            "definition": "Historical swapped x*60,y*80 diagnostic only; pN uses <= N px",
        },
    }


def software_metadata(device: torch.device) -> Dict[str, Any]:
    return {
        "python": platform.python_version(), "platform": platform.platform(),
        "torch": torch.__version__, "numpy": np.__version__, "opencv": cv2.__version__,
        "pytables": tables.__version__, "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "mps_available": bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()),
    }


def checkpoint_payload(
    model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int,
    validation_loss: float, args: argparse.Namespace, split: SplitDefinition,
    source: Mapping[str, Any], train_data: Mapping[str, Any],
    validation_data: Mapping[str, Any], loader_generator: torch.Generator,
    best_model_state_dict: Mapping[str, Tensor], best_epoch: int,
    best_validation_loss: float,
) -> Dict[str, Any]:
    rng_state: Dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "loader_generator": loader_generator.get_state(),
    }
    if torch.cuda.is_available():
        rng_state["torch_cuda_all"] = torch.cuda.get_rng_state_all()
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        rng_state["torch_mps"] = torch.mps.get_rng_state()
    return {
        "schema_version": 1,
        "checkpoint_role": "last_resume",
        "created_at_utc": utc_now(),
        "model_id": MODEL_ID,
        "model_scope": MODEL_SCOPE,
        "model_state_dict": model.state_dict(),
        "best_model_state_dict": dict(best_model_state_dict),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "validation_loss": validation_loss,
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation_loss,
        "seed": args.seed,
        "split": split.as_dict(),
        "protocol": protocol_metadata(),
        "source": dict(source),
        "data_fingerprints": {
            "train": train_data["effective_partition_sha256"],
            "validation": validation_data["effective_partition_sha256"],
        },
        "run_config": sanitised_run_config(args),
        "rng_state": rng_state,
    }


def clone_state_dict_to_cpu(state_dict: Mapping[str, Tensor]) -> Dict[str, Tensor]:
    return {name: value.detach().cpu().clone() for name, value in state_dict.items()}


def load_checkpoint(path: Path, device: torch.device) -> Dict[str, Any]:
    try:
        value = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        value = torch.load(path, map_location=device)
    if not isinstance(value, dict) or "model_state_dict" not in value:
        raise BenchmarkInputError(f"{path}: unsupported checkpoint")
    return value


def validate_checkpoint(checkpoint: Mapping[str, Any], split: SplitDefinition) -> None:
    expected = split.as_dict()
    actual = checkpoint.get("split", {})
    for field in (
        "split_id", "train_ids", "validation_ids", "train_list_sha256",
        "validation_list_sha256",
    ):
        if actual.get(field) != expected[field]:
            raise BenchmarkInputError(f"checkpoint {field} does not match {SPLIT_ID}")
    if checkpoint.get("model_id") != MODEL_ID:
        raise BenchmarkInputError("checkpoint is not the vanilla MyModel benchmark model")
    actual_protocol = checkpoint.get("protocol", {})
    expected_protocol = protocol_metadata()
    for field in ("protocol_id", "sequence_length", "train_stride", "validation_stride", "input_height", "input_width"):
        if actual_protocol.get(field) != expected_protocol[field]:
            raise BenchmarkInputError(f"checkpoint protocol mismatch: {field}")


def sanitised_run_config(args: argparse.Namespace) -> Dict[str, Any]:
    private = {"data_root", "label_root", "output_dir", "checkpoint", "resume", "handler", "command"}
    return {
        key: value for key, value in vars(args).items()
        if key not in private and isinstance(value, (str, int, float, bool, type(None)))
    }


def assess_protocol_compliance(config: Mapping[str, Any]) -> Dict[str, Any]:
    required = {
        "epochs": OFFICIAL_EPOCHS,
        "batch_size": OFFICIAL_BATCH_SIZE,
        "learning_rate": OFFICIAL_LEARNING_RATE,
        "seed": OFFICIAL_SEED,
        "deterministic": True,
        "overlap_check": "all",
    }
    mismatches = [
        f"{field}={config.get(field)!r} (official {expected!r})"
        for field, expected in required.items()
        if config.get(field) != expected
    ]
    return {
        "protocol_id": PROTOCOL_ID,
        "compliant": not mismatches,
        "status": "single_seed_reference" if not mismatches else "nonstandard_exploratory",
        "mismatches": mismatches,
    }


def ensure_output_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    names = {"checkpoint_best.pt", "checkpoint_last.pt", "history.jsonl", "results.json", "results.csv"}
    collisions = sorted(child.name for child in path.iterdir() if child.name in names)
    if collisions:
        raise BenchmarkInputError(
            "output directory contains prior benchmark files; choose a new directory: "
            + ", ".join(collisions)
        )


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


CSV_FIELDS = (
    "protocol_id", "model", "theta", "sequence_length", "train_stride",
    "validation_stride", "seed", "p3_percent", "p5_percent",
    "p10_percent", "parameters_millions", "flops_millions", "split_id",
    "train_list_sha256", "val_list_sha256", "checkpoint_sha256", "source_commit",
    "reproduction_status", "metric_id", "mean_error_px", "validation_loss",
    "best_epoch", "selected_epoch", "best_validation_loss", "epochs_requested",
    "batch_size", "learning_rate", "optimizer", "loss_function", "checkpoint_rule",
    "evaluated_points", "train_sequences", "validation_sequences", "parameter_count",
    "train_data_fingerprint_sha256", "validation_data_fingerprint_sha256",
    "source_tree_fingerprint_sha256", "legacy_swapped_metric_id",
    "legacy_swapped_p3_percent", "legacy_swapped_p5_percent",
    "legacy_swapped_p10_percent", "legacy_swapped_mean_error_px",
)


def write_results_csv(path: Path, result: Mapping[str, Any]) -> None:
    metrics = result["validation_metrics"]
    legacy = metrics.get("legacy_swapped_metric", {})
    training_config = (
        result["run_config"]
        if result["mode"] == "train"
        else result["training"].get("checkpoint_run_config", {})
    )
    row = {
        "protocol_id": PROTOCOL_ID, "model": MODEL_ID, "theta": "",
        "sequence_length": SEQUENCE_LENGTH, "train_stride": TRAIN_STRIDE,
        "validation_stride": VALIDATION_STRIDE,
        "seed": result["run_config"]["seed"], "p3_percent": metrics["p3_percent"],
        "p5_percent": metrics["p5_percent"], "p10_percent": metrics["p10_percent"],
        "parameters_millions": result["model"]["parameters"] / 1_000_000,
        "flops_millions": "", "split_id": SPLIT_ID,
        "train_list_sha256": result["split"]["train_list_sha256"],
        "val_list_sha256": result["split"]["validation_list_sha256"],
        "checkpoint_sha256": result["checkpoint"]["sha256"],
        "source_commit": result["source"]["source_commit"],
        "reproduction_status": result["protocol_compliance"]["status"],
        "metric_id": metrics["metric_id"],
        "mean_error_px": metrics["mean_error_px"], "validation_loss": metrics["loss"],
        "best_epoch": result["training"].get("best_epoch", ""),
        "selected_epoch": result["training"].get("best_epoch", ""),
        "best_validation_loss": result["training"].get("best_validation_loss", metrics["loss"]),
        "epochs_requested": training_config.get("epochs", ""),
        "batch_size": training_config.get("batch_size", result["run_config"]["batch_size"]),
        "learning_rate": training_config.get("learning_rate", ""),
        "optimizer": "Adam",
        "loss_function": "SmoothL1Loss(mean)",
        "checkpoint_rule": "minimum_validation_loss_first_tie",
        "evaluated_points": metrics["evaluated_points"],
        "train_sequences": len(EXPECTED_TRAIN_IDS) * len(window_starts(TRAIN_STRIDE)),
        "validation_sequences": len(EXPECTED_VALIDATION_IDS) * len(window_starts(VALIDATION_STRIDE)),
        "parameter_count": result["model"]["parameters"],
        "train_data_fingerprint_sha256": result["data"].get("train", {}).get("effective_partition_sha256", ""),
        "validation_data_fingerprint_sha256": result["data"]["validation"]["effective_partition_sha256"],
        "source_tree_fingerprint_sha256": result["source"]["source_tree_fingerprint_sha256"],
        "legacy_swapped_metric_id": legacy.get("metric_id", ""),
        "legacy_swapped_p3_percent": legacy.get("p3_percent", ""),
        "legacy_swapped_p5_percent": legacy.get("p5_percent", ""),
        "legacy_swapped_p10_percent": legacy.get("p10_percent", ""),
        "legacy_swapped_mean_error_px": legacy.get("mean_error_px", ""),
    }
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerow(row)


def build_result(
    mode: str, args: argparse.Namespace, split: SplitDefinition,
    source: Mapping[str, Any], device: torch.device, checkpoint_path: Path,
    metrics: Mapping[str, Any], val_data: Mapping[str, Any],
    inventory: Mapping[str, Any], model: nn.Module, training: Mapping[str, Any],
    train_data: Optional[Mapping[str, Any]] = None,
    checkpoint_training_source: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    data: Dict[str, Any] = {"validation": dict(val_data)}
    if train_data is not None:
        data["train"] = dict(train_data)
    result: Dict[str, Any] = {
        "schema_version": 1, "completed_at_utc": utc_now(), "mode": mode,
        "model": {"model_id": MODEL_ID, "scope": MODEL_SCOPE, "parameters": parameter_count(model)},
        "split": split.as_dict(), "recording_disjoint": True,
        "protocol": protocol_metadata(), "run_config": sanitised_run_config(args),
        "source": dict(source), "software": software_metadata(device),
        "checkpoint": {"filename": checkpoint_path.name, "sha256": sha256_path(checkpoint_path)},
        "validation_metrics": dict(metrics), "training": dict(training),
        "data": data, "input_inventory": dict(inventory),
    }
    compliance_config = (
        result["run_config"]
        if mode == "train"
        else training.get("checkpoint_run_config", {})
    )
    result["protocol_compliance"] = assess_protocol_compliance(compliance_config)
    if checkpoint_training_source is not None:
        result["checkpoint_training_source"] = dict(checkpoint_training_source)
    return result


def prepare_common(args: argparse.Namespace) -> Tuple[SplitDefinition, Dict[str, Any], torch.device, Dict[str, Any]]:
    split = load_and_validate_split()
    inventory = validate_input_inventory(args.data_root.resolve(), args.label_root.resolve(), split)
    device = select_device(args.device)
    seed_everything(args.seed, args.deterministic)
    source = git_source_metadata()
    print(f"split={SPLIT_ID} train=16 val=2 device={device}", flush=True)
    extras = inventory["train"]["extra_h5_ids_ignored"]
    if extras:
        print("ignoring non-official train H5 files: " + ", ".join(extras), flush=True)
    return split, inventory, device, source


def build_datasets(
    args: argparse.Namespace, split: SplitDefinition, include_train: bool
) -> Tuple[Optional[SEETWindowDataset], SEETWindowDataset]:
    train_dataset = None
    if include_train:
        train_dataset = SEETWindowDataset(
            split.train_ids, "train", args.data_root.resolve(), args.label_root.resolve(),
            args.overlap_check, args.overlap_samples,
        )
    val_dataset = SEETWindowDataset(
        split.validation_ids, "val", args.data_root.resolve(), args.label_root.resolve(),
        args.overlap_check, args.overlap_samples,
    )
    return train_dataset, val_dataset


def run_train(args: argparse.Namespace) -> int:
    if min(args.epochs, args.batch_size) <= 0 or args.learning_rate <= 0:
        raise BenchmarkInputError("epochs, batch-size, and learning-rate must be positive")
    if args.num_workers < 0 or args.overlap_samples < 0:
        raise BenchmarkInputError("worker/sample counts cannot be negative")
    if args.resume is not None and not args.resume.is_file():
        raise BenchmarkInputError(f"resume checkpoint does not exist: {args.resume}")
    ensure_output_directory(args.output_dir)
    split, inventory, device, source = prepare_common(args)
    train_dataset, val_dataset = build_datasets(args, split, True)
    assert train_dataset is not None
    train_data, val_data = train_dataset.metadata(), val_dataset.metadata()
    print(f"cached {len(train_dataset)} train and {len(val_dataset)} validation sequences", flush=True)

    loader_generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
        generator=loader_generator, worker_init_fn=seed_worker if args.num_workers else None,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = make_loader(
        val_dataset, args.batch_size, False, args.seed, args.num_workers, device
    )
    model = MyModel().to(device)
    criterion, optimizer = nn.SmoothL1Loss(), Adam(model.parameters(), lr=args.learning_rate)
    start_epoch, best_epoch, best_loss = 1, 0, float("inf")
    best_model_state: Optional[Dict[str, Tensor]] = None
    resume_lineage: Optional[Dict[str, Any]] = None

    if args.resume is not None:
        resumed = load_checkpoint(args.resume, device)
        validate_checkpoint(resumed, split)
        if resumed.get("checkpoint_role") != "last_resume":
            raise BenchmarkInputError(
                "only checkpoint_last.pt (role=last_resume) can be resumed; "
                "checkpoint_best.pt is evaluation-only"
            )
        previous_config = resumed.get("run_config", {})
        for field in ("seed", "batch_size", "learning_rate", "deterministic", "num_workers"):
            if previous_config.get(field) != getattr(args, field):
                raise BenchmarkInputError(
                    f"resume requires unchanged {field}: checkpoint={previous_config.get(field)!r}, "
                    f"CLI={getattr(args, field)!r}"
                )
        previous_sources = resumed.get("source", {}).get("source_file_sha256")
        if previous_sources != source.get("source_file_sha256"):
            raise BenchmarkInputError("resume runner/model source hashes differ from the checkpoint")
        resume_lineage = {
            "checkpoint_filename": args.resume.name,
            "checkpoint_sha256": sha256_path(args.resume),
            "checkpoint_epoch": int(resumed["epoch"]),
            "source": resumed.get("source", {}),
        }
        if resumed.get("data_fingerprints", {}).get("train") != train_data["effective_partition_sha256"]:
            raise BenchmarkInputError("resume checkpoint train-data fingerprint mismatch")
        if resumed.get("data_fingerprints", {}).get("validation") != val_data["effective_partition_sha256"]:
            raise BenchmarkInputError("resume checkpoint validation-data fingerprint mismatch")
        model.load_state_dict(resumed["model_state_dict"])
        optimizer.load_state_dict(resumed["optimizer_state_dict"])
        start_epoch = int(resumed["epoch"]) + 1
        best_loss = float(resumed.get("best_validation_loss", resumed["validation_loss"]))
        best_epoch = int(resumed.get("best_epoch", resumed["epoch"]))
        carried_best = resumed.get("best_model_state_dict")
        if carried_best is None:
            if int(resumed["epoch"]) != best_epoch:
                raise BenchmarkInputError(
                    "resume checkpoint predates best-state carryover and is not itself the best epoch"
                )
            carried_best = resumed["model_state_dict"]
        best_model_state = clone_state_dict_to_cpu(carried_best)
        rng = resumed.get("rng_state", {})
        if rng:
            random.setstate(rng["python"])
            np.random.set_state(rng["numpy"])
            torch.set_rng_state(rng["torch_cpu"].cpu())
            loader_generator.set_state(rng["loader_generator"].cpu())
            if "torch_cuda_all" in rng and torch.cuda.is_available():
                torch.cuda.set_rng_state_all([value.cpu() for value in rng["torch_cuda_all"]])
            if "torch_mps" in rng and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                torch.mps.set_rng_state(rng["torch_mps"].cpu())
        if start_epoch > args.epochs:
            raise BenchmarkInputError(
                f"resume checkpoint is epoch {start_epoch - 1}, but --epochs={args.epochs}"
            )
        print(f"resuming at epoch {start_epoch}", flush=True)

    best_path = args.output_dir / "checkpoint_best.pt"
    last_path = args.output_dir / "checkpoint_last.pt"
    history_path = args.output_dir / "history.jsonl"
    with history_path.open("w") as history:
        for epoch in range(start_epoch, args.epochs + 1):
            started = time.monotonic()
            print(f"epoch {epoch}/{args.epochs}", flush=True)
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_metrics = evaluate(
                model, val_loader, criterion, device, args.include_legacy_swapped_metric
            )
            improved = float(val_metrics["loss"]) < best_loss
            if improved:
                best_loss, best_epoch = float(val_metrics["loss"]), epoch
                best_model_state = clone_state_dict_to_cpu(model.state_dict())
            record = {
                "epoch": epoch, "training_loss": train_loss,
                "validation_metrics": val_metrics,
                "duration_seconds": time.monotonic() - started,
            }
            history.write(json.dumps(record, sort_keys=True) + "\n")
            history.flush()
            assert best_model_state is not None
            payload = checkpoint_payload(
                model, optimizer, epoch, float(val_metrics["loss"]), args, split, source,
                train_data, val_data, loader_generator, best_model_state, best_epoch,
                best_loss,
            )
            if resume_lineage is not None:
                payload["resume_lineage"] = resume_lineage
            torch.save(payload, last_path)
            if improved:
                best_payload = dict(payload)
                best_payload["checkpoint_role"] = "best_eval"
                best_payload["model_state_dict"] = best_model_state
                best_payload["epoch"] = best_epoch
                best_payload["validation_loss"] = best_loss
                best_payload.pop("best_model_state_dict", None)
                best_payload.pop("optimizer_state_dict", None)
                best_payload.pop("rng_state", None)
                torch.save(best_payload, best_path)
                print(f"saved best checkpoint at epoch {epoch}", flush=True)
            print(
                f"epoch {epoch}: train_loss={train_loss:.6f} "
                f"val_loss={val_metrics['loss']:.6f} p10={val_metrics['p10_percent']:.3f}%",
                flush=True,
            )

    # A resumed segment may never improve on the carried best.  In that case,
    # materialise the carried weights as this output directory's best file.
    if not best_path.exists():
        carried_payload = load_checkpoint(last_path, device)
        carried_payload["model_state_dict"] = best_model_state
        carried_payload["epoch"] = best_epoch
        carried_payload["validation_loss"] = best_loss
        carried_payload["checkpoint_role"] = "best_eval"
        carried_payload.pop("best_model_state_dict", None)
        carried_payload.pop("optimizer_state_dict", None)
        carried_payload.pop("rng_state", None)
        torch.save(carried_payload, best_path)
    best_checkpoint = load_checkpoint(best_path, device)
    validate_checkpoint(best_checkpoint, split)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    metrics = evaluate(model, val_loader, criterion, device, args.include_legacy_swapped_metric)
    result = build_result(
        "train", args, split, source, device, best_path, metrics, val_data, inventory,
        model, {"best_epoch": best_epoch, "best_validation_loss": best_loss,
                "history_filename": history_path.name, "last_checkpoint_filename": last_path.name,
                "resume_lineage": resume_lineage},
        train_data=train_data,
    )
    write_json(args.output_dir / "results.json", result)
    write_results_csv(args.output_dir / "results.csv", result)
    print(
        f"complete: p10={metrics['p10_percent']:.3f}% "
        f"checkpoint_sha256={result['checkpoint']['sha256']}", flush=True,
    )
    return 0


def run_eval(args: argparse.Namespace) -> int:
    if args.batch_size <= 0 or args.num_workers < 0 or args.overlap_samples < 0:
        raise BenchmarkInputError("invalid batch/worker/sample count")
    if not args.checkpoint.is_file():
        raise BenchmarkInputError(f"checkpoint does not exist: {args.checkpoint}")
    ensure_output_directory(args.output_dir)
    split, inventory, device, source = prepare_common(args)
    checkpoint = load_checkpoint(args.checkpoint, device)
    validate_checkpoint(checkpoint, split)
    checkpoint_seed = int(checkpoint.get("seed", args.seed))
    if args.seed != checkpoint_seed:
        print(
            f"evaluation uses checkpoint training seed {checkpoint_seed} "
            f"instead of CLI seed {args.seed}", flush=True,
        )
    args.seed = checkpoint_seed
    seed_everything(args.seed, args.deterministic)
    _, val_dataset = build_datasets(args, split, False)
    val_data = val_dataset.metadata()
    expected_fingerprint = checkpoint.get("data_fingerprints", {}).get("validation")
    if expected_fingerprint and expected_fingerprint != val_data["effective_partition_sha256"]:
        raise BenchmarkInputError("checkpoint validation-data fingerprint mismatch")
    loader = make_loader(val_dataset, args.batch_size, False, args.seed, args.num_workers, device)
    model = MyModel().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    metrics = evaluate(
        model, loader, nn.SmoothL1Loss(), device, args.include_legacy_swapped_metric
    )
    result = build_result(
        "eval", args, split, source, device, args.checkpoint, metrics, val_data,
        inventory, model, {
            "best_epoch": checkpoint.get("best_epoch", checkpoint.get("epoch")),
            "best_validation_loss": checkpoint.get(
                "best_validation_loss", checkpoint.get("validation_loss")
            ),
            "checkpoint_run_config": checkpoint.get("run_config", {}),
        },
        checkpoint_training_source=checkpoint.get("source", {}),
    )
    write_json(args.output_dir / "results.json", result)
    write_results_csv(args.output_dir / "results.csv", result)
    print(
        f"complete: p10={metrics['p10_percent']:.3f}% "
        f"checkpoint_sha256={result['checkpoint']['sha256']}", flush=True,
    )
    return 0


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-root", type=Path, required=True, help="root containing train/ and val/ H5 directories")
    parser.add_argument("--label-root", type=Path, required=True, help="directory containing ID.txt labels")
    parser.add_argument("--output-dir", type=Path, required=True, help="new/empty output directory")
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--overlap-check", choices=("sample", "all", "none"), default="sample",
        help="verify sampled (default), all, or no adjacent stride-1 H5 overlaps",
    )
    parser.add_argument("--overlap-samples", type=int, default=8)
    parser.add_argument(
        "--include-legacy-swapped-metric", action=argparse.BooleanOptionalAction,
        default=True, help="also report clearly labelled historical x*60,y*80 metric",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train/evaluate vanilla MyModel on recording-disjoint SEET-public-v1"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train")
    add_common_arguments(train)
    train.add_argument("--epochs", type=int, default=100)
    train.add_argument("--learning-rate", type=float, default=0.001)
    train.add_argument(
        "--resume", type=Path,
        help="checkpoint_last.pt to resume into a new output directory; best is evaluation-only",
    )
    train.set_defaults(handler=run_train)
    evaluate_parser = commands.add_parser("eval")
    add_common_arguments(evaluate_parser)
    evaluate_parser.add_argument("--checkpoint", type=Path, required=True)
    evaluate_parser.set_defaults(handler=run_eval)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except BenchmarkInputError as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
