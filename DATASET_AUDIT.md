# SEET data artifact and reproducibility audit

This audit separates the experimental protocol described in the 3ET paper
from the data artifacts that were later made public. Their recording counts do
not all match, so they must not share one unqualified label such as “the 3ET
split.”

Audit date: 2026-10-28.

## Artifact inventory

| Artifact | Observed contents | Split identity | Benchmark guidance |
| --- | --- | --- | --- |
| [3ET paper](https://arxiv.org/abs/2308.11771) | 22 recordings selected from LPW; 11k event frames; 80/20 data allocation | Exact recording IDs, allocation unit/procedure, and eye-speed calculation unrecovered | Cite as the paper protocol only |
| This repository at commit `477b85e468567def676e096834e0e936598429fe` | 16 train IDs, 2 validation IDs | `SEET-public-v1` | Official public dataset split going forward |
| [Google Drive event-frame artifact](https://drive.google.com/drive/folders/16qH_wv_oVNysJARtHIUrIXbHjOygfq_i) | 18 train recording directories, 2 validation recording directories, 62 label text files | `SEET-eventframes-v0` | Artifact inventory; do not call it the paper split |
| [Google Drive raw artifact](https://drive.google.com/drive/folders/1HeOS5YBLruzHjwMKyBQfVTc_mJbsy_R1) | 62 raw HDF5 files, 62 frame-time files, 51 AVI files, plus v2e metadata/output files | No train/validation split encoded | Suitable as source recordings; construct a declared split explicitly |
| [Historical Tonic download](https://dl.dropboxusercontent.com/s/1hyer8egd8843t9/ThreeET_Eyetracking.zip?dl=0) | 62 raw event HDF5 files and 62 label files; its bundled manifests select 16 train and 2 validation entries but only 17 unique IDs | `SEET-tonic-legacy-v0` | Raw data reusable; bundled split deprecated because of exact recording leakage |

The four LPW recording IDs not present in the audited raw artifact are
`subj_1_vid_4`, `subj_6_vid_2`, `subj_13_vid_9`, and `subj_16_vid_13`.
The absence of those files does not identify the 22 recordings used in the
paper.

Google Drive folder contents can change. The versioned manifests record the
inventory observed on the audit date. For archival experiments, retain your
own file listing and checksums alongside the results.

## Split differences

The event-frame Drive artifact has the same validation IDs as
`SEET-public-v1`. Its training partition adds two recordings:
`subj_8_vid_7` and `subj_9_vid_16`.

The historical Tonic split manifests substitute `subj_18_vid_7` for
`subj_18_vid_2` in training while also keeping `subj_18_vid_7` in validation.
This is complete-recording leakage, not merely a repeated subject. The raw
files themselves remain suitable for use with a corrected manifest such as
`SEET-public-v1`. The audited archive fingerprint is:

```text
bytes   3874426299
md5     b6c652b06fdfd85721f39e2dbe12f4e8
sha256  e0048e6c39c6b558d3880d6f62c683116ded3e91fc22920e29845108aa902a7d
```

## Raw and preprocessed pipelines are different

The two Google Drive links are not interchangeable:

- The **raw** artifact contains event streams and frame timestamps. Use an
  event loader (for example Tonic) or your own representation-generation code
  to create time bins or voxel/event frames.
- The **preprocessed event-frame** artifact contains derived HDF5 tensors and
  labels used by the legacy code path.
- `eyetracking-convlstm/process_event.py` is a legacy **sequence-windowing**
  step. It expects intermediate HDF5 files with a `/vector` array under the
  hard-coded `data_ts_500` path. It does not parse the raw v2e HDF5 format.
- `eyetracking-convlstm/convlstm-et-pytorch-event.py` expects windowed tensors
  in `./dataset/data_ts_pro/{train,val}` and labels in
  `./dataset/pupil_st/label`. Update the legacy paths or arrange the same
  directory layout before running it.

A reproducible result should record, at minimum: data artifact fingerprint or
inventory date, split ID, representation and time-bin settings, sequence
length and stride, source commit, random seed, checkpoint hash, and evaluation
metric implementation.
