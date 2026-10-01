# Point Group Classification with Linear Softmax Probes

This comparison measures how readily point-group labels can be decoded from frozen 32-dimensional crystal features. DPA features reduced by PCA achieve the highest test accuracy and macro-F1 among the four representations.

| Representation | Test accuracy | Test macro-F1 | Shuffled-label test accuracy |
|---|---:|---:|---:|
| DPA mean pooling → PCA32 | **45.94%** | **46.72%** | 15.51% |
| GMTNet | 29.84% | 24.07% | 17.13% |
| DPA-GMTNet | 27.33% | 20.05% | 18.32% |
| Global + PG fused features | 32.05% | 24.81% | 19.05% |

**Table 1.** Seven-class point-group classification on the curated reduced dielectric dataset, assembled from **Materials Project dielectric data released with DTNet** and **calculation-matched JARVIS dielectric data released with GMTNet**. The frozen split contains 5,001 training, 637 validation and 677 test structures. All representations have 32 dimensions; learned model features are extracted before the explicit symmetry mask. Higher accuracy and macro-F1 are better. The control permutes training labels only, using the same permutation across representations. Bold indicates the highest primary test metrics.

## Probe protocol

The backbone is frozen. A linear layer with softmax is fitted using mean cross-entropy plus L2 weight regularization; the intercept is unpenalized. Feature standardization uses training statistics only. The regularization strength is selected from seven fixed values between 1e-6 and 1 using validation macro-F1, with no test-set selection. DPA PCA is also fitted on the training set only and retains 57.00% of its total feature variance. All 56 fits, including shuffled-label controls and regularization candidates, passed convergence checks.

## Scope and limitations

GMTNet uses its historical epoch-93 checkpoint, an exception to the subsequently introduced epoch >100 rule. DPA-GMTNet and Global + PG use epoch 196. No new backbone training or checkpoint selection was performed.

These results measure linear decodability, not dielectric prediction quality or causal use of symmetry. Global + PG routing already uses point-group information, and symmetry-derived preprocessing can affect the representations. This is a single-split comparison, not a statistical significance test. Global + PG selects the smallest regularization value in the grid, so its probe is not exhaustively optimized.

Sources: [full probe results and reproducibility details](../src/probes/README.md), [dataset provenance](training_and_dataset.md), and [frozen dataset manifest](../data/manifests/curated_tensors_reduced_gt_5pct.json).
