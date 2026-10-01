# Frozen point-group probe

Run `python -m src.probes.pg_linear --output results/pg-probe/RUN --device cuda`
inside Slurm using the existing DPA4 environment. Reads the 192 frozen reduced-total
DPA feature shards and the SHA-verified reduced manifest/JSONL. Never runs DPA or
uses dielectric targets, symmetry masks, PG experts or PG metadata as model inputs.

Primary labels are the seven source point groups used to define the reduced dataset.
Cached current-PG agreement is reported separately; it does not silently change labels.
Inputs: arithmetic mean over atoms of all 3200 feature coordinates; controls are
mean even scalar channels, irrep-copy norms AFTER pooling (nonlinear invariant
descriptor, not a strictly linear probe of the original representation), atomic
fractions, and training-label permutation. The cache was computed in canonical frames:
raw-coordinate success does not establish orientation-independent symmetry recognition.

Classifier: affine multiclass ridge least squares on one-hot labels, argmax scores;
this is a linear probe, not a neural MLP or logistic regression. Train-only centering
and standardization, constant coordinates discarded, intercept unpenalized. Objective:
mean squared one-hot error + alpha * squared weight norm. Seven predeclared alphas
1e-6..1, selected on validation macro-F1 (then accuracy, then larger alpha).
Test labels never enter fitting or model selection. No train+validation refit.
Output: per-alpha train/validation metrics, selected held-out metrics, per-class recall,
confusion matrices, predictions, split counts, runtime and provenance. Single split/seed
is exploratory; composition/canonicalization and DPA pretraining overlap limit causal claims.
