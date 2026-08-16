# Brain Tumour MRI Classification — Multi-Model Comparison + Ensemble

Deep learning pipeline for 4-class brain tumour MRI classification (glioma, meningioma, no tumor, pituitary), comparing a custom CNN against three transfer-learning backbones, with a soft-voting ensemble on top.

## Dataset

[Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/vasavajaiminiben/brain-tumor-detection-image) (Kaggle, publicly available, de-identified). 32,331 training images / 1,810 official held-out test images across 4 classes.

## Models

| Model | Type | Notebook |
|---|---|---|
| TumorDetNet | Custom residual CNN w/ Squeeze-and-Excitation blocks, trained from scratch | [`Notebooks/brain-tumor-tumordetnet-v6-resumable.ipynb`](Notebooks/brain-tumor-tumordetnet-v6-resumable.ipynb) |
| ResNet50 | ImageNet-pretrained, 2-phase fine-tuning | [`Notebooks/brain-tumor-resnet50-v6-resumable.ipynb`](Notebooks/brain-tumor-resnet50-v6-resumable.ipynb) |
| InceptionV3 | ImageNet-pretrained, 2-phase fine-tuning | [`Notebooks/brain-tumor-inceptionv3-v6-resumable.ipynb`](Notebooks/brain-tumor-inceptionv3-v6-resumable.ipynb) |
| MobileNetV2 | ImageNet-pretrained, 2-phase fine-tuning | [`Notebooks/brain-tumor-mobilenetv2-v6-resumable.ipynb`](Notebooks/brain-tumor-mobilenetv2-v6-resumable.ipynb) |
| **Ensemble** | Soft-voting combination of all 4 above, using TTA-averaged probabilities per model | [`Notebooks/brain-tumor-ensemble-v3-with-figures.ipynb`](Notebooks/brain-tumor-ensemble-v3-with-figures.ipynb) |

## Protocol

- **Leakage-free split**: test set is the dataset's official held-out `testing/` folder (never carved from `training/`); train/val is an 85/15 stratified split of `training/` only. Identical protocol across all 4 models for a fair comparison.
- **Evaluation**: per-class precision/recall/F1, sensitivity/specificity, Cohen's kappa, bootstrap 95% confidence intervals, test-time augmentation (10 passes), McNemar's test for statistical significance, expected calibration error (ECE), ablation studies.
- Seed = 42 throughout.

## Results (test set)

Single-pass vs. TTA vs. ensemble, all configurations:

| Configuration | Accuracy | Macro F1 | Cohen's κ |
|---|---|---|---|
| TumorDetNet (single-pass) | **0.9895** (best individual result) | 0.9893 | 0.9858 |
| TumorDetNet (TTA) | 0.9878 | 0.9873 | 0.9835 |
| ResNet50 (single-pass) | 0.9779 | 0.9762 | 0.9701 |
| ResNet50 (TTA) | 0.9862 | 0.9853 | 0.9813 |
| InceptionV3 (single-pass) | 0.9669 | 0.9650 | 0.9551 |
| InceptionV3 (TTA) | 0.9790 | 0.9776 | 0.9716 |
| MobileNetV2 (single-pass) | 0.9105 | 0.9087 | 0.8790 |
| MobileNetV2 (TTA) | 0.9674 | 0.9652 | 0.9559 |
| **Ensemble (TTA, equal-weight)** | **0.9867** | **0.9856** | **0.9820** |
| Ensemble (TTA, accuracy-weighted) | 0.9867 | 0.9856 | 0.9820 |

Full ensemble headline metrics (equal-weight soft voting, TTA N=10):

| Metric | Value |
|---|---|
| Accuracy | 0.9867 |
| Mean Specificity | 0.9956 |
| Macro F1 | 0.9856 |
| Weighted F1 | 0.9867 |
| Macro AUC | 0.9996 |
| mAP | 0.9993 |
| Cohen's κ | 0.9820 |

Bootstrap 95% CI (ensemble): Accuracy [0.9812, 0.9923], Macro F1 [0.9792, 0.9914], Macro AUC [0.9989, 1.0000].

Full per-model CSVs (per-class metrics, sensitivity/specificity, bootstrap CI, McNemar's test, ablation, calibration) are in `brain_tumor_ensemble_outputs/`.

### Key finding: ensembling did not significantly beat the best single model

TumorDetNet's single-pass accuracy (0.9895) is the highest individual result in this study. The 4-model soft-voting ensemble reached 0.9867 — very close, with strong calibration (Macro AUC 0.9996, mAP 0.9993) and robustness — but McNemar's test comparing the ensemble against TumorDetNet alone was **not statistically significant (statistic = 10.0, p = 0.832)**. This suggests TumorDetNet had already reached close to the achievable ceiling on this dataset, leaving little room for the ensemble to correct further. Both results are reported transparently rather than only the more favorable one. The ensemble's real value lies in de-risking architecture selection and improving worst-case robustness across models, not in a guaranteed accuracy gain.

## Repo structure

```
Notebooks/                     — all 5 Jupyter notebooks (4 individual models + ensemble)
brain_tumor_ensemble_outputs/  — extracted results CSVs + figures for quick reference / paper tables
model_comparison_summary.csv
```

<!--
## Status
- [x] 4 backbone models trained and evaluated
- [x] Ensemble notebook built and run (TTA-averaged soft-voting, v3 with figures)
- [x] Ensemble results finalized — see Results table above and `brain_tumor_ensemble_outputs/`
- [ ] SOTA literature comparison table populated with cited baselines
- [ ] Cross-dataset generalization check
- [ ] Manuscript write-up
-->
