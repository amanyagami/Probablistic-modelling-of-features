# Probabilistic Modelling of Features

## Quick start (`featprob`)

`featprob` is a small, tested package for class-conditional GMM modelling of deep features
and feature-space OOD scoring. Reproducible environment via [uv](https://docs.astral.sh/uv/):

```bash
uv sync                      # core + dev tools (numpy, scipy, scikit-learn, matplotlib, pytest, ruff)
uv sync --extra torch        # + torch/torchvision for feature extraction
uv sync --extra viz          # + umap-learn (optional: --extra opentsne)
uv run pytest                # run the tests
uv run python notebooks/00_quickstart.py   # end-to-end demo on synthetic data
```

```python
import numpy as np
from featprob import ClassConditionalGMM

# X_train: (n, d) features, y_train: labels; X_cal: held-out ID features (NOT the training set)
model = ClassConditionalGMM(max_components=4, random_state=0).fit(X_train, y_train)
model.calibrate(X_cal)
scores = model.anomaly_score(X_test, method="mahalanobis")   # large => anomalous
metrics = model.evaluate(X_id_test, X_ood, method="loglik")  # auroc, aupr_in, aupr_out, fpr_at_95tpr
```

Features from a PyTorch model: `featprob.features.FeatureExtractor(model).extract(loader)`
(hooks the input of the final `Linear` layer; torch is imported lazily). Seeded t-SNE/UMAP
helpers are in `featprob.viz`. Before committing notebooks run `uv run nbstripout --install`
(or use pre-commit) so outputs are not committed.

## What the notebooks do / known issues

The notebooks are research scratchpads. They need files that are **not in this repo** (`models/`
with `ResNet34`/`DenseNet3` having an `intermediate_forward`, `FS.py` fault injector,
`pre_trained/*.pth`, `adv_samples/*.pt`, `data/` with CIFAR-10, SVHN, iSUN, LSUN, DTD,
CIFAR-100, precomputed `extracted_features_*.pkl`), hardcode `cuda:0`/`cuda:1`, and use RAPIDS
(`cuml`, `cudf`) and `cupy` in several cells. They cannot be re-run from this repo alone.

- `GMM_Resnet.ipynb`: CIFAR-10 ResNet34 (10 classes). Loads pre-extracted features of 5 layers, fits **one
  10-component GMM per layer on PCA(200) of the train features** (not class-conditional), scores ID
  (CIFAR-10 test), PGD, FGSM, AutoAttack, LSUN, SVHN, iSUN with `score_samples`, writes
  `gmm_scores11thdec.csv`, and uses the 5th percentile of ID scores as a 95%-TPR threshold, then
  reports accuracy/AUROC/AUPR per layer.
- `visualizations.ipynb`, `visualize only two classes.ipynb`, `visualize_features_deeply.ipynb`: global-average-pooled features
  per layer (DenseNet3/ResNet34), t-SNE (sklearn or cuML) of ID vs. adversarial vs. OOD sets with
  hand-made label offsets (+10/+100/...), per-class running mean/precision (Mahalanobis-style
  distance, then the same 5th-percentile/AUROC analysis on `results/*.csv`).
- `try to detect SDC.ipynb` (repo root): "SDC" = silent data corruption, i.e. a hardware/bit-flip fault
  injected into a random layer (via the missing `FS` module) that silently flips the model prediction.
  It picks 32 random neurons per module, records per-class min/max/mean activation on 2000 train
  images, scores test images by distance of the neuron activation to class means, and checks whether
  the score flags images whose prediction changed under injection.

Bugs / leakage / pitfalls found by reading the code (not by running it):

1. GMM notebook: **PCA is re-fit on every test set** (`pca.fit_transform` on ID/OOD/adversarial features) instead
   of `transform` with the PCA fitted on train features, so each set lives in a different basis and the
   GMM scores are not comparable. This alone can invalidate the reported AUROC/AUPR.
2. GMM notebook: the threshold is the 5th percentile of the *same* ID test scores that are then used to compute
   accuracy/TPR (no separate calibration split; TPR is 95% by construction). A later duplicated cell references
   undefined `umap_model` / `gmm_final['ID_Train']`.
3. Features are of the ResNet34 model, but adversarial samples are named `densenet3_cifar10_*.pt` (if they were crafted against
   that DenseNet they are transfer attacks; the notebooks do not say) and the SDC model/data pairing is inconsistent across notebooks.
4. Adversarial samples keep the original (true) labels and are loaded as raw `float` arrays, so a normalisation
   mismatch with the clean pipeline (not verifiable here) would itself look like "OOD".
5. Mahalanobis stats: the rolling covariance update inverts the stored precision each batch (numerically fragile,
   uses an ad-hoc 1e-6 ridge), and the test-time loop uses `class_features = features` for every class and
   then `break`s after one batch.
6. t-SNE cells: plots break after a fixed number of samples (`len(features_all) > 2000` counts batches vs.
   samples inconsistently), perplexity up to 600 on 2000 points, a 3-D plot mixes `X_label_100_sampled` into the
   label-200 scatter, and `features_array` is used without being recomputed (stale variable).
7. SDC notebook: `normalized_score[neuron] = [...]` overwrites rather than appends (only the last class
   counts); `range(1, 10)` over `means[neuron][i]` where keys were built with `enumerate(..., 1)` shifts class
   ids so class 9 is never used; SDC indices are 0-based but detections use `val + 1`, so
   detected/false-positive counts are misaligned by one; thresholds differ between detected (1.0) and false
   positives (0.3); the first 500 test images calibrate the score and the rest evaluate it, but the test set
   is also what defines `SDC`; forward hooks are registered on every module and never removed; random neuron
   selection is unseeded.
8. Metric hygiene: "accuracy" mixes ID and OOD at one threshold; no seeds are fixed for sampling in plots;
   results CSV `gmm_scores11thdec.csv` (8 MB) is committed and notebooks carry large outputs.

---

## Original README

Exploratory notebooks for probabilistic and geometric analysis of deep feature representations,
with emphasis on class-wise structure and adversarial robustness.

---

## Visual gallery

<div align="center">

| CIFAR-100 t-SNE | Different Classes |
|:---------------:|:-----------------:|
| <img src="images/Cifar100_TSNE.png" alt="CIFAR-100 t-SNE" width="350"/> | <img src="images/Different_Classes.png" alt="Different Classes" width="350"/> |

<br/>

<img src="images/Final_LAyer_TSNE.png" alt="Final Layer t-SNE" width="700"/>

**Figure:** Final-layer embedding geometry of the trained model.

</div>

---

## Notebooks

- `notebooks/visualize only two classes.ipynb` — focused two-class embedding analysis.  
- `notebooks/visualize_features_deeply.ipynb` — multi-class and adversarial analysis.  
- `notebooks/visualizations.ipynb` — additional plots.  
- `notebooks/GMM_Resnet.ipynb` — GMM / model experiments.

## How to run

Interactive (recommended):
```bash
uv sync   # legacy list kept in requirements.legacy.txt
jupyter lab
# then open a notebook and run cells
