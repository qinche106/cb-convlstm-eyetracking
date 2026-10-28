#!/usr/bin/env python3
"""Validate versioned SEET recording splits and their provenance metadata."""

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


RECORDING_RE = re.compile(r"^subj_([0-9]+)_vid_([0-9]+)$")
ROLE_TO_COUNT = {
    "train": "train_recordings",
    "validation": "validation_recordings",
}


def natural_recording_key(recording_id: str) -> Tuple[int, int]:
    match = RECORDING_RE.fullmatch(recording_id)
    if match is None:
        return (sys.maxsize, sys.maxsize)
    return (int(match.group(1)), int(match.group(2)))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_recordings(path: Path) -> Tuple[List[str], List[str]]:
    errors: List[str] = []
    recordings: List[str] = []
    seen = set()

    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        recording_id = raw_line.strip()
        if not recording_id:
            continue
        if RECORDING_RE.fullmatch(recording_id) is None:
            errors.append(
                "{}:{}: invalid recording ID {!r}".format(
                    path, line_number, recording_id
                )
            )
        if recording_id in seen:
            errors.append(
                "{}:{}: duplicate recording ID {!r}".format(
                    path, line_number, recording_id
                )
            )
        seen.add(recording_id)
        recordings.append(recording_id)

    return recordings, errors


def subjects(recordings: Iterable[str]) -> set:
    result = set()
    for recording_id in recordings:
        match = RECORDING_RE.fullmatch(recording_id)
        if match is not None:
            result.add(str(int(match.group(1))))
    return result


def sorted_recordings(recordings: Iterable[str]) -> List[str]:
    return sorted(recordings, key=natural_recording_key)


def check_expected_set(
    manifest_path: Path,
    field: str,
    actual: set,
    expected_value: Optional[Sequence[str]],
    errors: List[str],
) -> None:
    if expected_value is None:
        return
    expected = set(expected_value)
    if actual != expected:
        errors.append(
            "{}: {} is {}, expected {}".format(
                manifest_path,
                field,
                sorted(actual),
                sorted(expected),
            )
        )


def validate_manifest(manifest_path: Path) -> Dict[str, Any]:
    errors: List[str] = []
    warnings: List[str] = []
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "manifest": str(manifest_path),
            "split_id": manifest_path.parent.name,
            "status": "invalid",
            "errors": ["{}: {}".format(manifest_path, exc)],
            "warnings": [],
        }

    split_id = manifest.get("split_id", manifest_path.parent.name)
    status = manifest.get("status", "unknown")
    if manifest.get("schema_version") != 1:
        errors.append("{}: schema_version must be 1".format(manifest_path))
    if split_id != manifest_path.parent.name:
        errors.append(
            "{}: split_id {!r} must match directory name {!r}".format(
                manifest_path, split_id, manifest_path.parent.name
            )
        )

    files = manifest.get("files")
    if files is None:
        if status != "unrecovered":
            errors.append(
                "{}: only an unrecovered split may omit list files".format(
                    manifest_path
                )
            )
        return {
            "manifest": str(manifest_path),
            "split_id": split_id,
            "status": status,
            "train_recordings": None,
            "validation_recordings": None,
            "unique_recordings": None,
            "recording_overlap": None,
            "subject_overlap": None,
            "errors": errors,
            "warnings": warnings,
        }

    role_recordings: Dict[str, List[str]] = {}
    for role in ("train", "validation"):
        file_metadata = files.get(role)
        if not isinstance(file_metadata, dict):
            errors.append(
                "{}: files.{} must be an object".format(manifest_path, role)
            )
            continue
        relative_path = file_metadata.get("path")
        if not isinstance(relative_path, str):
            errors.append(
                "{}: files.{}.path must be a string".format(manifest_path, role)
            )
            continue
        list_path = manifest_path.parent / relative_path
        if not list_path.is_file():
            errors.append("{}: missing {} list {}".format(manifest_path, role, list_path))
            continue

        recordings, list_errors = load_recordings(list_path)
        errors.extend(list_errors)
        role_recordings[role] = recordings

        declared_sha256 = file_metadata.get("sha256")
        actual_sha256 = sha256(list_path)
        if declared_sha256 != actual_sha256:
            errors.append(
                "{}: {} SHA-256 is {}, expected {}".format(
                    manifest_path, role, actual_sha256, declared_sha256
                )
            )

        declared_count = manifest.get("counts", {}).get(ROLE_TO_COUNT[role])
        if declared_count != len(recordings):
            errors.append(
                "{}: {} count is {}, expected {}".format(
                    manifest_path, role, len(recordings), declared_count
                )
            )

    if set(role_recordings) != {"train", "validation"}:
        return {
            "manifest": str(manifest_path),
            "split_id": split_id,
            "status": status,
            "errors": errors,
            "warnings": warnings,
        }

    train = role_recordings["train"]
    validation = role_recordings["validation"]
    recording_overlap = set(train) & set(validation)
    subject_overlap = subjects(train) & subjects(validation)
    unique_recordings = len(set(train) | set(validation))

    declared_unique = manifest.get("counts", {}).get("unique_recordings")
    if declared_unique != unique_recordings:
        errors.append(
            "{}: unique recording count is {}, expected {}".format(
                manifest_path, unique_recordings, declared_unique
            )
        )

    integrity = manifest.get("integrity", {})
    check_expected_set(
        manifest_path,
        "recording overlap",
        recording_overlap,
        integrity.get("expected_recording_overlap"),
        errors,
    )
    check_expected_set(
        manifest_path,
        "subject overlap",
        subject_overlap,
        integrity.get("expected_subject_overlap"),
        errors,
    )

    if integrity.get("recording_disjoint_required") is True and recording_overlap:
        errors.append(
            "{}: recording-disjoint split contains {}".format(
                manifest_path, sorted_recordings(recording_overlap)
            )
        )
    if integrity.get("subject_disjoint_required") is True and subject_overlap:
        errors.append(
            "{}: subject-disjoint split contains subjects {}".format(
                manifest_path, sorted(subject_overlap, key=int)
            )
        )
    if recording_overlap:
        warnings.append(
            "complete recording overlap: {}".format(
                ", ".join(sorted_recordings(recording_overlap))
            )
        )

    aliases = manifest.get("compatibility_aliases", {})
    for role, alias_relative_path in aliases.items():
        if role not in role_recordings:
            errors.append(
                "{}: compatibility alias has unknown role {!r}".format(
                    manifest_path, role
                )
            )
            continue
        alias_path = manifest_path.parent / alias_relative_path
        if not alias_path.is_file():
            errors.append(
                "{}: missing compatibility alias {}".format(
                    manifest_path, alias_path
                )
            )
            continue
        alias_recordings, alias_errors = load_recordings(alias_path)
        errors.extend(alias_errors)
        if alias_recordings != role_recordings[role]:
            errors.append(
                "{}: compatibility alias {} differs from the versioned {} list".format(
                    manifest_path, alias_path, role
                )
            )

    return {
        "manifest": str(manifest_path),
        "split_id": split_id,
        "status": status,
        "train_recordings": len(train),
        "validation_recordings": len(validation),
        "unique_recordings": unique_recordings,
        "recording_overlap": sorted_recordings(recording_overlap),
        "subject_overlap": sorted(subject_overlap, key=int),
        "errors": errors,
        "warnings": warnings,
    }


def discover_manifests(repo_root: Path) -> List[Path]:
    return sorted((repo_root / "splits").glob("*/manifest.json"))


def print_human(results: Sequence[Dict[str, Any]]) -> None:
    for result in results:
        outcome = "FAIL" if result["errors"] else "OK"
        if result.get("train_recordings") is None:
            counts = "manifest only"
        else:
            counts = "train={} validation={} unique={}".format(
                result["train_recordings"],
                result["validation_recordings"],
                result["unique_recordings"],
            )
        print(
            "{} {} [{}]: {}".format(
                outcome, result["split_id"], result["status"], counts
            )
        )
        if result.get("recording_overlap") is not None:
            overlap = result["recording_overlap"] or "none"
            print("  recording overlap: {}".format(overlap))
            print("  subject overlap: {}".format(result["subject_overlap"] or "none"))
        for warning in result["warnings"]:
            print("  WARNING: {}".format(warning))
        for error in result["errors"]:
            print("  ERROR: {}".format(error))


def main(argv: Optional[Sequence[str]] = None) -> int:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "manifests",
        nargs="*",
        type=Path,
        help="manifest.json files (default: all versioned manifests)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit the complete audit result as JSON",
    )
    args = parser.parse_args(argv)

    manifest_paths = args.manifests or discover_manifests(repo_root)
    if not manifest_paths:
        parser.error("no split manifests found")

    results = [validate_manifest(path.resolve()) for path in manifest_paths]
    if args.json:
        print(json.dumps(results, indent=2, sort_keys=True))
    else:
        print_human(results)

    return 1 if any(result["errors"] for result in results) else 0


if __name__ == "__main__":
    sys.exit(main())
