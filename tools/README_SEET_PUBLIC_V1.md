# SEET-public-v1 vanilla ConvLSTM runner

`run_seet_public_v1.py` trains/evaluates the repository's existing vanilla
`MyModel` ConvLSTM. It does not implement the paper's change-based execution.

The runner accepts only the frozen `SEET-public-v1` 16/2 recording lists. It
checks their byte hashes and recording-level disjointness, reports and ignores
extra H5 files, and fingerprints every selected H5, label file, effective
frame cache, checkpoint, and runner source.

The selected preprocessed artifact is identified as `SEET-eventframes-v0`.
Every selected raw H5 and label file is also hashed, so the run does not rely
on the artifact name alone.

It reconstructs each training recording once from its first 40-frame window
plus each subsequent window's last frame (500 unique frames). Validation's 12
non-overlapping windows yield the 480 frames used by the legacy
`chunk=500,stride=40` pipeline. Recording order is lexicographic, matching the
legacy `sorted(...)` loader. Use `--overlap-check all` for an official run.

## Train

```bash
python tools/run_seet_public_v1.py train \
  --data-root /path/to/data_ts_pro \
  --label-root /path/to/labels \
  --output-dir /path/to/new-run-directory \
  --device auto \
  --epochs 100 \
  --batch-size 16 \
  --learning-rate 0.001 \
  --seed 1 \
  --overlap-check all
```

The data root contains `train/*.h5` and `val/*.h5`, each with `/vector`; the
label root contains `subj_<id>_vid_<id>.txt`. Outputs are
`checkpoint_best.pt`, `checkpoint_last.pt`, `history.jsonl`, `results.json`,
and `results.csv`. Nonstandard settings remain usable but are labelled
`nonstandard_exploratory`, never `single_seed_reference`.

Resume a checkpoint into another new output directory by adding:

```bash
--resume /path/to/old-run/checkpoint_last.pt
```

The runner requires matching source hashes and fixed training configuration,
restores CPU/CUDA/MPS and loader RNG state, and carries the prior best weights.
Only `checkpoint_last.pt` is resumable; `checkpoint_best.pt` intentionally
contains evaluation weights without optimizer/RNG state.

## Evaluate

```bash
python tools/run_seet_public_v1.py eval \
  --data-root /path/to/data_ts_pro \
  --label-root /path/to/labels \
  --checkpoint /path/to/run/checkpoint_best.pt \
  --output-dir /path/to/new-evaluation-directory \
  --device auto
```

The primary metric is Euclidean pixel error with `x*80,y*60` and strict
`distance < threshold` for P3/P5/P10. The separately named legacy diagnostic
uses the historical swapped `x*60,y*80` scaling.

## Released reference artifacts

The verified single-seed run is published under
[`results/SEET-public-v1-convlstm-code-v1/`](../results/SEET-public-v1-convlstm-code-v1/),
with its canonical row in
[`results/seet_public_v1_results.csv`](../results/seet_public_v1_results.csv).
It trained from scratch for 100 epochs on Snellius job `28705332` from clean
source commit `f2e36e9abbc029d1bf292f7755783009ccca404f`; independent job
`28709602` reloaded the selected epoch-83 checkpoint and reproduced every
primary and legacy metric exactly.

The release reports validation-only results from one seed on 2 recordings
(960 points). It is a vanilla public-code ConvLSTM reference, not a CB-ConvLSTM
execution result or a reconstruction of the paper's unrecovered 22-recording
experiment. PyTorch 2.1.2 also warned that CUDA
`max_pool3d_with_indices_backward` has no deterministic implementation, so the
artifact is not claimed to be bitwise reproducible across hardware or software
stacks. Exact metrics, fingerprints, hashes, and software versions are in the
[results registry](../results/README.md).

## Test

```bash
python -m unittest discover -s tests -p 'test_seet_public_v1_runner.py' -v
```
