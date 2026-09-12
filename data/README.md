# Tensor benchmark data

Downloaded for the global dielectric/elastic scope in `proposal.md`, with a
separate atom-resolved JARVIS-DFPT Born effective charge (BEC) preparation
path for the expanded implementation goal.

## Raw datasets

- raw/jarvis_gmtnet/jarvis_diele_piezo.pkl: calculation-matched JARVIS
  dielectric/piezoelectric release from the official GMTNet repository. The
  first-stage task uses only its dielectric field.
- raw/jarvis_gmtnet/jarvis_elastic.pkl: calculation-matched JARVIS elastic
  release from the official GMTNet repository.
- raw/matten/crystal_elasticity_tensor.json: MatTen v1.0.0 full dataset from
  Zenodo DOI 10.5281/zenodo.8190849, including its published split field.
- raw/jarvis_dfpt/raw_file_index.zip: JARVIS-DFT raw calculation index from
  Figshare DOI 10.6084/m9.figshare.13154159.v1. Its frozen DFPT section has
  5,000 calculation archives (about 9.95 GiB), while the associated paper
  reports 5,015 calculations. This release difference is recorded explicitly
  in `manifests/jarvis_dfpt_bec.json`.

The pinned source repositories are retained under sources/GMTNet and
sources/matten so that the original filtering, split, tensor convention and
training code remain auditable.

## Manifests

Run the following command after copying or re-downloading any raw file:

    python data/build_manifest.py

The generated files under manifests contain checksums, sample counts, filters,
and exact split membership. GMTNet uses its published seed 32 and 80/10/10
protocol after task filtering. MatTen split membership comes directly from the
downloaded dataset.

For elastic, direct comparison with the 14,220-sample GMTNet table additionally
requires the official second-stage structural-symmetry screen from
`GMTNet_elast/data.py`; the initial magnitude screen alone leaves 14,480 records
and is not benchmark-equivalent. `data/build_manifest.py --only elastic
--output-dir ...` reproduces the structure-derived forbidden-component mask,
rejects labels whose forbidden entries reach `1e-4 GPa`, and stores each accepted
6x6 support mask as a compact 36-bit integer. On Guqq this batch operation must
use `slurm/build_jarvis_elastic_manifest.sbatch` and write a candidate under
ignored `results/`; production manifests are promoted only from local Git.

The GMTNet files are Python pickles. Only load them when their SHA-256 values
match the manifests and their provenance is the pinned official repository.

## BEC extraction

The full BEC label must come from the same `vasprun.xml` as the structure; the
summary properties in the usual JARVIS JSON are not an atom-resolved BEC
dataset. A resumable extraction is provided by:

    python data/prepare_jarvis_bec.py \
      --index-zip data/raw/jarvis_dfpt/raw_file_index.zip \
      --output data/processed/jarvis_dfpt_bec.jsonl \
      --archive-dir data/raw/jarvis_dfpt/archives \
      --errors data/processed/jarvis_dfpt_bec_errors.jsonl \
      --workers 8

Each output record stores the calculation-matched final lattice, fractional
coordinates, element order, complete `N x 3 x 3` BEC tensor in elementary
charge, upstream archive MD5, `vasprun.xml` SHA-256, and acoustic-sum-rule
residual. Full extraction is a batch-processing task and must be submitted via
Slurm on the configured server.
