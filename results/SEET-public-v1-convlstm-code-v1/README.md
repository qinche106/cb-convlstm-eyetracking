# `SEET-public-v1-convlstm-code-v1` artifact record

This directory preserves the released vanilla public-code ConvLSTM reference
for the prospective official `SEET-public-v1` 16/2 split. The split is
recording-disjoint but not subject-disjoint. It is official for public results
reported from this repository going forward and is not a reconstruction of the
paper's unrecovered 22-recording experiment.

## Result

| Seed | P3 | P5 | P10 | Mean error | Validation loss | Best epoch | Parameters |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 87.500% | 98.958333% | 99.687500% | 1.6302475 px | 0.000206203724 | 83 | 416,882 |

The primary metric uses corrected `x*80,y*60` Euclidean pixel distance with
strict `distance < threshold`. These are validation-only measurements on 2
recordings (24 sequences, 960 points) from one seed. They are not held-out test
results, uncertainty estimates, or results for a change-based execution model.

## Files and hashes

| File | SHA-256 |
| --- | --- |
| `checkpoint_best.pt` | `d5a177d4d395f481e8b36441ec9e854e8488a6f11d047393916cb40cb1b09523` |
| `results.json` | `c918fad3c13a89eb04f4644fce778d7034258180fdaefb8f34abf6a552e1a4d6` |
| `results.csv` | `c6617b85e8c1ab213990cbcefa0450fbc9d2d6a560dc5e72df306040ae7b33b4` |
| `history.jsonl` | `e4168b75a86aaa239fab942115364428cc744345f98c39c1be4b75a6c8ac462b` |
| `independent_eval_results.json` | `4a87f5870416e543d4a729065a49e68610864bc78347054a82a1d51520c5f234` |
| `independent_eval_results.csv` | `e012b3cf3208bd52cbd9b65a1e79a49e5316e13e457e594b51703ae927a245c0` |

`results.json` and `results.csv` are the canonical training-produced records.
The independently generated files reload and evaluate the same best
checkpoint; their complete primary and legacy metric objects match the
canonical record exactly. Evaluation intentionally loads only the validation
partition, so its metadata does not repeat the training-data fingerprint.

## Provenance and limitations

- Clean source commit:
  `f2e36e9abbc029d1bf292f7755783009ccca404f`.
- Training: Snellius job `28705332`, NVIDIA A100 40 GB, 100 fresh epochs,
  exit code 0, elapsed 02:03:01.
- Independent evaluation: Snellius job `28709602`, exit code 0, empty stderr.
- Train/validation sequences: 7,376/24; evaluated validation points: 960.
- Train data fingerprint:
  `84632ef45b610cdb229971f0c7fe7ebd8ea48613b85603136d726765aa9cf151`.
- Validation data fingerprint:
  `522ddc9306a99c2a4866b89f6f13ccc1936864e5bd4f552727af0d6fb682f9eb`.
- Software: Python 3.11.3, PyTorch 2.1.2 with CUDA 12.1, NumPy 1.25.1,
  OpenCV 4.8.1, and PyTables 3.9.2.

PyTorch warned that CUDA `max_pool3d_with_indices_backward` has no
deterministic implementation even though deterministic algorithms were
requested with `warn_only=True`. Consequently this is a single-seed reference,
not a guarantee of bitwise-identical results on another hardware or software
stack.
