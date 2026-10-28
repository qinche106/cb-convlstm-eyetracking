# SEET split registry

Recording lists in this directory are immutable, versioned artifacts. New work
should name the exact split ID in papers, configuration files, and result
tables.

| Split ID | Train / validation | Recording-disjoint? | Intended use |
| --- | ---: | :---: | --- |
| `SEET-public-v1` | 16 / 2 | Yes | Recommended legacy public-code benchmark |
| `SEET-eventframes-v0` | 18 / 2 | Yes | Inventory of the legacy Google Drive event-frame folders |
| `SEET-tonic-legacy-v0` | 16 / 2 (17 unique) | **No** | Historical audit only; deprecated for evaluation |
| `SEET-paper-2023-unrecovered` | 22 selected, membership unknown | Unknown | Paper protocol record; exact lists not recovered |

## Benchmark policy

`SEET-public-v1` freezes the two list files that were already present at
`eyetracking-convlstm/train_files.txt` and
`eyetracking-convlstm/val_files.txt`. Those root-level files remain in place as
compatibility aliases for the training code.

The benchmark requires **complete recording IDs** to be disjoint between train
and validation. It is not subject-disjoint: subject 18 contributes different
recordings to both partitions. Results must therefore be described as
recording-disjoint, not participant-independent.

The paper reports 22 selected videos and an 80/20 data allocation, but the
exact recording IDs, allocation unit and procedure, and eye-speed selection
calculation have not been recovered.
`SEET-public-v1` must not be presented as the original paper split.

## Validation

From the repository root, run:

```bash
python3 tools/validate_splits.py
```

The validator checks ID syntax, duplicate IDs, declared file hashes and counts,
recording/subject overlap, and compatibility with the legacy root lists. Use
`--json` to save a machine-readable audit report.

Do not edit a released split in place. Add a new directory and increment its
version when changing membership.
