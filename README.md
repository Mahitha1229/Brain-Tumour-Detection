# Brain Tumour MRI Classification — Multi-Model Comparison + Ensemble

Deep learning pipeline for 4-class brain tumour MRI classification (glioma, meningioma, no tumor, pituitary), comparing a custom CNN against three transfer-learning backbones, with a soft-voting ensemble on top.

## Dataset
[Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/vasavajaiminiben/brain-tumor-detection-image) (Kaggle, publicly available, de-identified). 32,331 training images / 1,810 official held-out test images across 4 classes.

## Models
| Model | Type | Notebook |
|---|---|---|
| TumorDetNet | Custom residual CNN w/ Squeeze-and-Excitation blocks, trained from scratch | [`notebooks/brain-tumor-tumordetnet-v6-resumable.ipynb`](notebooks/brain-tumor-tumordetnet-v6-resumable.ipynb) |
| ResNet50 | ImageNet-pretrained, 2-phase fine-tuning | [`notebooks/brain-tumor-resnet50-v6-resumable.ipynb`](notebooks/brain-tumor-resnet50-v6-resumable.ipynb) |
| InceptionV3 | ImageNet-pretrained, 2-phase fine-tuning | [`notebooks/brain-tumor-inceptionv3-v6-resumable.ipynb`](notebooks/brain-tumor-inceptionv3-v6-resumable.ipynb) |
| MobileNetV2 | ImageNet-pretrained, 2-phase fine-tuning | [`notebooks/brain-tumor-mobilenetv2-v6-resumable.ipynb`](notebooks/brain-tumor-mobilenetv2-v6-resumable.ipynb) |
| **Ensemble** | Soft-voting combination of all 4 above, using TTA-averaged probabilities per model | [`notebooks/brain-tumor-ensemble-v2.ipynb`](notebooks/brain-tumor-ensemble-v2.ipynb) |

## Protocol
- **Leakage-free split**: test set is the dataset's official held-out `testing/` folder (never carved from `training/`); train/val is an 85/15 stratified split of `training/` only. Identical protocol across all 4 models for a fair comparison.
- **Evaluation**: per-class precision/recall/F1, sensitivity/specificity, Cohen's kappa, bootstrap 95% confidence intervals (1,000 resamples), test-time augmentation (10 passes), McNemar's test for statistical significance, expected calibration error (ECE), ablation studies.
- Seed = 42 throughout.

## Results (test set, TTA-boosted)

| Model | Accuracy | Macro F1 | Weighted F1 | Macro AUC | mAP | Mean Specificity |
|---|---|---|---|---|---|---|
| TumorDetNet (custom) | 0.9878 | 0.9873 | 0.9878 | 0.9995 | 0.9987 | 0.9963 |
| ResNet50 | 0.9840 | 0.9826 | 0.9839 | 0.9996 | 0.9964 | 0.9926 |
| InceptionV3 | 0.9757 | 0.9741 | 0.9756 | 0.9987 | 0.9940 | 0.9890 |
| MobileNetV2 | 0.9652 | 0.9632 | 0.9650 | 0.9974 | 0.9674 | 0.9703 |
| **Ensemble** | *see `results/ensemble_results_summary.csv` once generated* | | | | | |

Full per-model CSVs (per-class metrics, sensitivity/specificity, bootstrap CI, McNemar's test, ablation, calibration) are produced directly by each notebook's Output & Packaging section.

## Repo structure
```
notebooks/   — all 5 Jupyter notebooks (4 individual models + ensemble)
results/     — extracted results CSVs for quick reference / paper tables
```
<!--
## Status
- [x] 4 backbone models trained and evaluated
- [x] Ensemble notebook built (v1: raw single-pass; v2: adds TTA per model before combining)
- [ ] Ensemble results finalized on Kaggle
- [ ] SOTA literature comparison table populated with cited baselines
- [ ] Cross-dataset generalization check
- [ ] Manuscript write-up
-->
