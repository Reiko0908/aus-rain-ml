# Next-Day Rain Prediction (weatherAUS) — Traditional ML Pipeline on Tabular Data

Course assignment, **Task 1: Machine Learning with Tabular Data**.

| | |
|---|---|
| **Course** | Machine Learning |
| **Course code** | `CO3117` |
| **Semester / academic year** | Semester I, 2026–2027 |
| **University** | Faculty of Computer Science and Engineering, Ho Chi Minh City University of Technology (HCMUT), VNU-HCM |
| **Instructor** | `Dr.Truong Vinh Lan` |

## Group members

| Full name | Student ID | Email |
|---|---|---|
| `Le Tran Minh Phuong` | `2453025` | `phuong.letranminh@hmcut.edu.vn` |
| `Nguyen Le Duc Phu` | `2352918` | `phu.nguyen23022005@hcmut.edu.vn` |
| `Nguyen Hoang Thong` | `2353148` | `thong.nguyenkallidus@hcmut.edu.vn` |
| `Trương Anh Khoa ` | `2452566` | `khoa.truong2805@hcmut.edu.vn` |

## Objectives

- Build a **configurable traditional ML pipeline** for tabular data covering EDA, preprocessing (missing-value imputation, categorical encoding, outlier handling), scaling, dimensionality reduction, model training and evaluation.
- Compare **scalers** (StandardScaler, MinMaxScaler with different `feature_range`), **PCA** (none / 90% / 95% retained variance) and **models** (Logistic Regression, RBF-kernel SVM, CatBoost).
- Compare the traditional pipeline with one **deep learning pipeline** (MLP with learned categorical embeddings) — bonus extension.
- Save extracted feature representations (`.h5`, `.npy`) and report all results reproducibly.

**Dataset:** Rain in Australia (`weatherAUS`) — 145,460 daily observations from 49 weather stations (2007–2017), published by the Australian Bureau of Meteorology and packaged in the R package `rattle` / on Kaggle. Binary target `RainTomorrow` (22.4% rainy after removing 3,267 unlabelled rows). It has **missing values in 21 of 23 columns** (61% of the rows have at least one gap, in numeric and categorical columns) and **5 categorical columns** (`Location` with 49 levels, three wind directions, `RainToday`). Public download link used by the notebook:
<https://raw.githubusercontent.com/acakin/weatherAUS/master/weatherAUS.csv>

**Chronological split (70 / 15 / 15).** Consecutive days of one station are correlated, and the label of day *t* is the feature `RainToday` of day *t+1*, so a random split leaks neighbouring days. The oldest 70% of the days train, the next 15% select models, the newest 15% give the final numbers. A random split would have inflated macro-F1 by 0.016–0.025 (see report, Section 5.1).

## Results (test set = newest 15% of the days, 21,329 days; baseline accuracy 76.2%)

| Model | Pipeline | Accuracy | Macro-F1 | ROC-AUC | Recall (rain) |
|---|---|---|---|---|---|
| Logistic Regression | iterative imputation → one-hot → StandardScaler → no PCA, C=1 | 84.1% | 0.760 | 0.867 | 0.55 |
| RBF-SVM | median/"Missing" → quantile clipping → one-hot → MinMax(0,1) → PCA 95% (92 dims), C=1, balanced, 20k subsample | 78.5% | 0.738 | 0.855 | 0.76 |
| **CatBoost** | native categoricals, no scaling, no PCA, depth 8, 500 trees | **85.6%** | **0.784** | **0.893** | 0.59 |
| MLP with embeddings (deep learning) | quantile transform + learned category embeddings, 256-128 | 85.3% | 0.783 | 0.887 | 0.60 |
| LR on MLP hidden features | 128-d learned features → Logistic Regression | 85.0% | 0.779 | 0.886 | 0.59 |

Key findings: CatBoost and the MLP are essentially tied and clearly ahead of LR and the SVM; PCA lowered macro-F1 for every model while compressing only 20–23%; iterative imputation was the most useful preprocessing change; one-hot beat ordinal and target encoding; the SVM's low accuracy comes from `class_weight="balanced"`, which was chosen on validation macro-F1 and makes it flag many dry days as rain (precision 0.53).

Full analysis: [`reports/261_MachineLearning_Group10_Report.pdf`](reports/261_MachineLearning_Group10_Report.pdf).

## How to run

### Google Colab (recommended)
1. Open the notebook: `https://colab.research.google.com/drive/17wknJhTfAxLXYbOUoGstz2TAj5MkEylA?usp=sharing` (or upload `notebooks/Task1_Tabular_WeatherAUS.ipynb` to Colab).
2. `Runtime → Run all`.

The first code cell clones this repository (set `REPO_URL` to this repo's URL), installs `catboost` if missing, and the data is downloaded automatically from the public link above — **no Google Drive mounting is needed**.
A full run takes about 35 minutes on a 2-core CPU runtime; no GPU is needed. Set `FAST_MODE = True` for a ~7-minute smoke run on subsampled data (numbers will then differ from the report).

### Locally
```bash
git clone https://github.com/Reiko0908/aus-rain-ml && cd aus-rain-ml
pip install -r requirements.txt
jupyter notebook notebooks/Task1_Tabular_WeatherAUS.ipynb   # then Run All
```

### Library requirements
Python ≥ 3.10, numpy, pandas ≥ 2.0, scikit-learn ≥ 1.4, matplotlib, h5py, PyTorch ≥ 2.0, CatBoost ≥ 1.2 (see `requirements.txt`).

## Project structure

```
.
├── notebooks/
│   └── Task1_Tabular_WeatherAUS.ipynb   # Colab front-end: runs everything end to end
├── modules/
│   ├── data_loader.py      # download, cleaning, feature engineering, chronological 70/15/15 split
│   ├── eda.py              # EDA tables and figures
│   ├── preprocessing.py    # configurable imputer / outlier clipper / encoder / scaler / PCA
│   ├── models.py           # Logistic Regression, RBF-SVM, CatBoost (encoded + native categorical)
│   ├── experiment.py       # config -> pipeline, grid runner, CV tuning, feature saving (.h5/.npy)
│   ├── configs.py          # experiment definitions (scaler/PCA runs, ablations, tuning spaces)
│   ├── tabular_mlp.py      # MLP with categorical embeddings in PyTorch (deep learning pipeline)
│   ├── evaluation.py       # metrics and evaluation plots
│   └── plot_style.py       # shared figure style
├── features/               # extracted feature matrices (.h5 / .npy) + trained MLP
├── reports/
│   ├── 261_MachineLearning_Group10_Report.pdf   # final report
│   ├── figures/            # all figures used in the report
│   └── results/            # every experiment result as CSV / JSON
├── requirements.txt
└── README.md
```

### Feature files (`features/`)

| File | Content |
|---|---|
| `lr_features.h5` | Logistic Regression pipeline output: imputed, one-hot encoded, standardised matrix (train / val / test) |
| `svm_pca_*.npy` (also `svm_features.h5` on GitHub) | RBF-SVM pipeline output: clipped, MinMax-scaled, PCA-reduced matrix |
| `mlp_hidden_features.h5` | MLP learned representation (128-d last hidden layer) for train / val / test |
| `mlp_embeddings.pt` | Trained MLP: weights, preprocessing state and training history |

CatBoost is run in native categorical mode, so it has no engineered feature matrix to save.
Each `.h5` file contains `X_train, y_train, X_val, y_val, X_test, y_test` and the feature names as an attribute:
```python
import h5py
with h5py.File("features/lr_features.h5") as f:
    X_train, y_train = f["X_train"][:], f["y_train"][:]
```

## Data source
Australian Bureau of Meteorology, Climate Data Online (<http://www.bom.gov.au/climate/data>); table distributed with the R package `rattle` (G. Williams) and on Kaggle ("Rain in Australia").

## Links
- Report: [`reports/261_MachineLearning_Group10_Report.pdf`](reports/261_MachineLearning_Group10_Report.pdf)
- Colab notebook: [`Colab Notebook`](https://colab.research.google.com/drive/17wknJhTfAxLXYbOUoGstz2TAj5MkEylA?usp=sharing)
