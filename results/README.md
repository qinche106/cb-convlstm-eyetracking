# Official results registry

The official numeric table in this repository must report results evaluated on
`SEET-public-v1`. The paper's Table II values are intentionally not copied here:
the paper reports a different, incompletely specified protocol, so relabeling
those values as 16/2 results would create false provenance.

`seet_public_v1_results.template.csv` defines the table columns but contains no
measurements. Replace the template with `seet_public_v1_results.csv` only after
the values can be tied to the exact 16/2 lists. Every published row must include:

- the `SEET-public-v1` split ID and train/validation list hashes;
- model variant, threshold, temporal sequence length, and random seed;
- detection rates and the exact metric implementation;
- source commit and checkpoint SHA-256;
- data artifact fingerprint, representation settings, and preprocessing
  configuration in a companion metadata file.

The versioned vanilla ConvLSTM reference protocol and runner are documented in
[`tools/README_SEET_PUBLIC_V1.md`](../tools/README_SEET_PUBLIC_V1.md). Its
primary metric uses the corrected 80x60 geometry (`x*80`, `y*60`) and strict
`distance < threshold`; the historical swapped-axis metric is diagnostic only.

The plot under `eyetracking-convlstm/plot/event_plot_28.png` is retained as a
historical qualitative example. It is not a complete reproducibility record
and must not be interpreted as a verified 16/2 benchmark run.

Do not populate the final table from memory or from paper numbers alone. If a
checkpoint or evaluation log cannot establish the 16/2 membership, rerun the
evaluation before publishing a value.
