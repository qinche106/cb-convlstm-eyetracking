# Official results registry

The canonical table is
[`seet_public_v1_results.csv`](seet_public_v1_results.csv). It contains only
measurements tied to the prospective official public split
`SEET-public-v1`. The split is recording-disjoint but not subject-disjoint and
is official for results reported from this repository going forward. It is not
claimed to reconstruct the paper's unrecovered 22-recording experiment. The
paper's Table II values are intentionally not copied or relabeled here.

## Released reference row

| Protocol | Model | Seed | P3 | P5 | P10 | Mean error | Validation loss | Best epoch |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `SEET-public-v1-convlstm-code-v1` | vanilla public-code ConvLSTM | 1 | 87.500% | 98.958333% | 99.687500% | 1.6302475 px | 0.000206203724 | 83 |

This is a validation-only, single-seed reference: 2 validation recordings,
24 non-overlapping sequences, and 960 evaluated points. It is not a held-out
test result, a multi-seed estimate, or a result for the paper's change-based
execution mechanism. The model has 416,882 parameters.

The primary metric is corrected 80x60 geometry (`x*80`, `y*60`) with strict
`distance < threshold`. The historical swapped-axis diagnostic is kept only
for audit: P3 90.104167%, P5 99.062500%, P10 99.895833%, and mean error
1.5721422 px.

## Artifact bundle

[`SEET-public-v1-convlstm-code-v1/`](SEET-public-v1-convlstm-code-v1/)
contains:

| File | SHA-256 | Purpose |
| --- | --- | --- |
| [`checkpoint_best.pt`](SEET-public-v1-convlstm-code-v1/checkpoint_best.pt) | `d5a177d4d395f481e8b36441ec9e854e8488a6f11d047393916cb40cb1b09523` | selected epoch-83 weights |
| [`results.json`](SEET-public-v1-convlstm-code-v1/results.json) | `c918fad3c13a89eb04f4644fce778d7034258180fdaefb8f34abf6a552e1a4d6` | canonical training result and full provenance |
| [`results.csv`](SEET-public-v1-convlstm-code-v1/results.csv) | `c6617b85e8c1ab213990cbcefa0450fbc9d2d6a560dc5e72df306040ae7b33b4` | canonical generated result row |
| [`history.jsonl`](SEET-public-v1-convlstm-code-v1/history.jsonl) | `e4168b75a86aaa239fab942115364428cc744345f98c39c1be4b75a6c8ac462b` | all 100 fresh training epochs |
| [`independent_eval_results.json`](SEET-public-v1-convlstm-code-v1/independent_eval_results.json) | `4a87f5870416e543d4a729065a49e68610864bc78347054a82a1d51520c5f234` | independent best-checkpoint evaluation metadata |
| [`independent_eval_results.csv`](SEET-public-v1-convlstm-code-v1/independent_eval_results.csv) | `e012b3cf3208bd52cbd9b65a1e79a49e5316e13e457e594b51703ae927a245c0` | independent generated result row |

`checkpoint_last.pt` is intentionally not published because it is not needed
to verify or use the selected reference weights.

## Execution and verification

- Source: clean commit
  `f2e36e9abbc029d1bf292f7755783009ccca404f`.
- Training: Snellius job `28705332`, one NVIDIA A100 40 GB GPU, 100 epochs
  from scratch, exit code 0, elapsed 02:03:01.
- Independent evaluation: Snellius job `28709602`, exit code 0, empty stderr.
  It reloaded `checkpoint_best.pt` into a new process and reproduced the full
  primary and legacy metric objects exactly.
- Split-list hashes: train
  `d0a3e202fcd084f55bcd265aca0abfeadb33d90b8989af08fc47a27eb6aeaf50`;
  validation
  `32a21bfb5105f89568ff266e3a0e9d6aa54fb5d5aa1b56775360ac76de4d400e`.
- Effective data fingerprints: train
  `84632ef45b610cdb229971f0c7fe7ebd8ea48613b85603136d726765aa9cf151`;
  validation
  `522ddc9306a99c2a4866b89f6f13ccc1936864e5bd4f552727af0d6fb682f9eb`.
- Data counts: 7,376 training sequences; 24 validation sequences; 960
  validation points. `history.jsonl` contains epochs 1 through 100 with no
  resume lineage.
- Software recorded in `results.json`: Python 3.11.3, PyTorch 2.1.2, CUDA
  12.1, NumPy 1.25.1, OpenCV 4.8.1, and PyTables 3.9.2.

PyTorch warned that CUDA `max_pool3d_with_indices_backward` lacks a
deterministic implementation. Deterministic algorithms were requested with
`warn_only=True`, but this warning means the release is a single-seed reference
and not a claim of bitwise reproducibility across hardware or software stacks.

The versioned vanilla ConvLSTM reference protocol and runner are documented in
[`tools/README_SEET_PUBLIC_V1.md`](../tools/README_SEET_PUBLIC_V1.md). Its
primary metric uses the corrected 80x60 geometry (`x*80`, `y*60`) and strict
`distance < threshold`; the historical swapped-axis metric is diagnostic only.

The plot under `eyetracking-convlstm/plot/event_plot_28.png` is retained as a
historical qualitative example. It is not a complete reproducibility record
and must not be interpreted as this verified 16/2 benchmark run.
