# Results registry

`paper_table_ii.csv` is a machine-readable transcription of Table II in the
3ET paper. The metadata file records its source and reproduction status. These
values were **reported by the paper, not rerun on `SEET-public-v1`**; the
paper's exact 22-recording membership remains unrecovered.

The plot under `eyetracking-convlstm/plot/event_plot_28.png` is retained as a
historical qualitative example. It is not a complete reproducibility record
and should not be interpreted as a fresh benchmark run.

For a new run, add a new result file instead of overwriting published values.
Record the split ID, list-file hashes, data artifact fingerprint, source commit,
preprocessing and representation settings, sequence length/stride, seed,
checkpoint hash, and metric implementation. This keeps future
`SEET-public-v1` results directly auditable.
