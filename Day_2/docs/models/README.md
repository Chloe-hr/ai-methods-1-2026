---
license: mit
library_name: sklearn
pipeline_tag: tabular-regression
tags:
  - sklearn
  - tabular-regression
  - real-estate
  - built-environment
  - gradient-boosting
  - iaac-maai
  - ai-methods-1
metrics:
  - r2
  - mae
  - mape
---

# House price estimator (India): predicting the asking price of a flat from its characteristics

This model estimates the **asking price of a flat in India** (in ₹) from its size, typology, location, position in the building, orientation, views and a few other characteristics. It is a scikit-learn pipeline (imputation + encoding + `HistGradientBoostingRegressor`) trained on about 50,000 cleaned real-estate listings from 81 Indian cities.

It is the **worked example** for the Day 2 assignment of **AI Methods I: Machine Learning for the Built Environment** (IAAC MaAI 2026/27): it shows students what a published model and its model card should look like.

- **GitHub repository (full pipeline):** https://github.com/iaac-maai/ai-methods-1-2026 (notebooks `Day_1/docs/data_cleaning_and_EDA.ipynb` and `Day_2/docs/application_of_supervised_learning.ipynb`)
- **Authors:** AI Methods I course team, IAAC

## Intended use

- **Intended users and use:** students, architects and designers who want a **quick, rough estimate** of the market value of a flat in an early feasibility study, or who want to see how a design parameter (size, number of bedrooms, floor, orientation) moves the estimate. It is a **teaching model**.
- **Out-of-scope uses:**
  - Any real financial, valuation, lending or investment decision.
  - Cities that are not in the training data, countries other than India, and building types other than flats (no houses, villas or land).
  - Values outside the training range: asking prices outside roughly ₹14–650 Lakh, or flats outside roughly 28–370 m² of carpet area. The model cannot extrapolate beyond what it has seen.
  - Estimating **sale** prices: the training data contains **asking** prices.

## Model details

| | |
|---|---|
| **Problem type** | Regression |
| **Target** | `log10(asking price in ₹)` |
| **Algorithm** | `HistGradientBoostingRegressor` (gradient-boosted decision trees) |
| **Key hyperparameters** | `learning_rate=0.1`, `max_leaf_nodes=63`, `max_iter=300`, `min_samples_leaf=20`, `random_state=42` (tuned with `GridSearchCV`, see below) |
| **Preprocessing** | Numeric: median imputation. Nominal categories: most-frequent imputation + one-hot encoding (`handle_unknown="ignore"`). `furnishing`: ordinal encoding (Unfurnished < Semi-Furnished < Furnished, unknown = −1). 20 input columns become 115 features. |
| **File** | `model.joblib`: the whole fitted scikit-learn `Pipeline` (preprocessing + model), saved with `joblib` |
| **Library versions** | Python 3.12, scikit-learn 1.9.1, pandas 3.0.6, numpy 2.5.3, joblib 1.6.0 |

## Input features

The model expects a `pandas.DataFrame` with **exactly these 20 columns**. Missing values (`NaN`) are allowed in the numeric columns; they are imputed with the training median. Unknown categories are allowed; they are ignored by the one-hot encoder.

| Column | Type | Unit / values | Description | Preprocessing |
|---|---|---|---|---|
| `location` | category | 81 Indian cities, e.g. `"Bangalore"`, `"New Delhi"` | City of the listing | one-hot |
| `bhk` | integer | count | Number of bedrooms (BHK = bedroom, hall, kitchen) | median imputation |
| `property_type` | category | `"Flat"` | Type of property (only flats in the training data) | one-hot |
| `carpet_area` | float | sqft (1 m² = 10.7639 sqft) | Usable (net) floor area | median imputation |
| `super_area` | float | sqft | Gross area: carpet area + walls + share of common areas | median imputation |
| `efficiency` | float | 0.3–1 | Net-to-gross ratio: `carpet_area / super_area` | median imputation |
| `bathroom` | integer | count | Number of bathrooms | median imputation |
| `balcony` | integer | count | Number of balconies | median imputation |
| `balcony_was_missing` | 0 / 1 | flag | 1 if the balcony count was not given in the listing | median imputation |
| `floor_num` | float | floor (0 = ground, negative = basement) | Floor of the flat | median imputation |
| `total_floors` | float | floors | Number of floors of the building | median imputation |
| `floor_ratio` | float | 0–1 | `floor_num / total_floors` (0 = ground, 1 = top floor) | median imputation |
| `facing` | category | `East`, `North`, `North-East`, `North-West`, `South`, `South-East`, `South-West`, `West`, `Unknown` | Orientation of the main façade | one-hot |
| `view_garden` | 0 / 1 | flag | The flat overlooks a garden or park | median imputation |
| `view_pool` | 0 / 1 | flag | The flat overlooks a pool | median imputation |
| `view_main_road` | 0 / 1 | flag | The flat overlooks a main road | median imputation |
| `transaction` | category | `New Property`, `Resale`, `Other`, `Unknown` | New build or resale | one-hot |
| `furnishing` | category | `Unfurnished`, `Semi-Furnished`, `Furnished`, `Unknown` | Furnishing level | ordinal encoding |
| `ownership` | category | `Freehold`, `Leasehold`, `Co-operative Society`, `Power Of Attorney`, `Unknown` | Legal tenure | one-hot |
| `parking_spaces` | float | count | Number of parking spaces | median imputation |

## Output

The model returns **`log10` of the asking price in Indian rupees (₹)**. Convert it back:

- price in ₹ = `10 ** prediction`
- price in Lakh = `10 ** prediction / 100_000` (1 Lakh = ₹100,000; 1 Crore = ₹10,000,000)

## Training data

| | |
|---|---|
| **Dataset** | House Price (India), by Juhi Bhojani |
| **Source** | [kaggle.com/datasets/juhibhojani/house-price](https://www.kaggle.com/datasets/juhibhojani/house-price) |
| **Licence** | Community Data License Agreement – Sharing 1.0 (CDLA-Sharing-1.0) |
| **Origin** | Listings scraped from an Indian real-estate portal (dataset last updated August 2023) |
| **One row is** | One listing of a flat for sale |
| **Size** | 187,531 raw rows → 62,460 clean listings → 49,968 for training, 12,492 for testing |
| **Coverage** | 81 Indian cities, roughly the same number of listings per city (about 2,500–2,900 in the largest ones); flats only |

**Cleaning** (Day 1 notebook): removed 119,339 exact duplicates hidden by a row counter; dropped empty, constant and free-text columns; converted prices ("42 Lac", "1.40 Cr") and areas (sqft, sqyrd, sqm) to numbers; split `"10 out of 11"` into floor and building height; extracted the number of bedrooms from the listing title; recovered missing gross areas from `price / price_per_sqft`; removed listings without a price and the most extreme 1% on each side (price, price per sqft, carpet area); filled missing bathrooms and balconies with the median per number of bedrooms.

**Leakage:** the column `price_per_sqft` is computed from the price itself, so it was **removed** from the inputs (with it, a linear model reaches a meaningless R² of 1.000).

## Training procedure

- **Split:** 80 / 20 train / test, `random_state=42`. The test set was used once, at the end.
- **Validation:** 5-fold cross-validation (`KFold`, shuffled, `random_state=42`) on the training set, with all preprocessing inside the pipeline so nothing leaks between folds.
- **Models compared** (same folds, same preprocessing; R² on `log10(price)`):

| Model | CV R² (mean ± std) | Training time per fold |
|---|---|---|
| Baseline (always predicts the mean) | 0.000 ± 0.000 | 0.1 s |
| KNN (k=10) | 0.723 ± 0.002 | 0.1 s |
| Decision tree (max_depth=10) | 0.744 ± 0.004 | 0.5 s |
| Linear regression | 0.782 ± 0.003 | 1.2 s |
| Ridge | 0.782 ± 0.003 | 0.2 s |
| Gradient boosting (default settings) | 0.843 ± 0.003 | 2.4 s |
| Random forest (100 trees) | 0.849 ± 0.003 | 35.7 s |

- **Tuning:** `GridSearchCV` over `learning_rate` ∈ {0.05, 0.1, 0.2} × `max_leaf_nodes` ∈ {15, 31, 63}, `max_iter=300` (45 fits). Best: `learning_rate=0.1`, `max_leaf_nodes=63`, CV R² **0.857**.
- **Why this model:** random forest and gradient boosting were tied within noise before tuning, but gradient boosting trains about 15× faster, handles missing values natively and improved further with tuning.

## Evaluation

Results on the held-out **test set** (12,492 flats, used once):

| Metric | Test | Cross-validation |
|---|---|---|
| R² on `log10(price)` | **0.854** | 0.857 |
| MAE | **₹19.4 Lakh** | – |
| RMSE | ₹34.8 Lakh | – |
| MAPE (average error in %) | **21.1%** | – |
| Flats predicted within ±20% | **59%** | – |

For comparison, the baseline that always predicts the average price is off by ₹50.7 Lakh (62%) on average. The training R² is 0.894: a moderate gap to the test score, so the model fits the training data somewhat better than new data but is not severely overfitting.

![Predicted vs actual price and distribution of errors on the test set](regression_test_results.png)

**Where it performs worst:** large and very heterogeneous markets, where the neighbourhood matters as much as the city, which the model does not know. Median absolute error per city (cities with ≥ 100 test flats): best in Surat (11%), Raipur (11%) and Coimbatore (12%); worst in **Mumbai (24%), Thane (24%) and Chennai (25%)**.

**What the model relies on** (permutation importance on the test set): mostly **gross area** (`super_area`) and **city** (`location`), then carpet area and building height. Orientation, views and floor position add very little.

![Permutation importance](feature_importance.png)

## Limitations and biases

- **Asking prices, not sale prices:** they reflect what sellers hope to get.
- **Sampling bias:** the scraper collected roughly the same number of listings per city, regardless of city size. The model compares cities well but does not represent the Indian market as a whole.
- **Missing information:** no neighbourhood, building age, construction quality, amenities or exact location. These explain much of the remaining error, especially in large cities.
- **Self-reported and incomplete data:** carpet area is missing in 44% of the listings and parking in 57%; they are imputed with the median, which hides real variation. Listings with missing details tend to be cheaper, and the model partly learns this ("Unknown" categories).
- **Correlated features:** size variables (`carpet_area`, `super_area`, `bhk`, `bathroom`) overlap strongly, so changing one of them alone in a "what if" scenario can give unrealistic combinations and unreliable estimates.
- **Cleaning decisions shape the model:** the 1st–99th percentile filter removes very cheap and very expensive listings, so the model is not reliable for luxury or very low-cost flats.
- **Time:** the data is from 2023 or earlier; prices change.
- **Importance is not causation:** a feature the model uses is not necessarily a cause of the price.

## How to use

Requirements: `pip install scikit-learn==1.9.1 pandas joblib huggingface_hub`

```python
import joblib
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

path = hf_hub_download(repo_id="iaac-maai/day2-example-house-price", filename="model.joblib")
model = joblib.load(path)

# A new 2 BHK flat of 85 m² net in Bangalore, 5th floor of a 12-storey building
carpet_sqft = 85 * 10.7639
example = pd.DataFrame([{
    "location": "Bangalore", "bhk": 2, "property_type": "Flat",
    "carpet_area": carpet_sqft, "super_area": carpet_sqft / 0.8, "efficiency": 0.8,
    "bathroom": 2, "balcony": 1, "balcony_was_missing": 0,
    "floor_num": 5, "total_floors": 12, "floor_ratio": 5 / 12,
    "facing": "East", "view_garden": 1, "view_pool": 0, "view_main_road": 0,
    "transaction": "New Property", "furnishing": "Semi-Furnished", "ownership": "Freehold",
    "parking_spaces": 1,
}])

log_price = model.predict(example)[0]
print(f"Estimated asking price: ₹{10 ** log_price / 1e5:,.0f} Lakh")
```

⚠️ **Security:** `.joblib` files are Python pickles and can execute code when loaded. Only load models from sources you trust. Use **scikit-learn 1.9.1**: the model may not load, or may behave differently, with other versions.

## Reproducibility

The full pipeline is in the course repository https://github.com/iaac-maai/ai-methods-1-2026:

1. `Day_1/docs/data_cleaning_and_EDA.ipynb` downloads the raw data from Kaggle and produces `house_prices_clean.csv`.
2. `Day_2/docs/application_of_supervised_learning.ipynb` trains, compares, tunes, evaluates and saves this model (section 4).

The packages and exact versions are in the repository's `requirements.txt`. All random steps use `random_state=42`.

## Authors and licence

AI Methods I course team, Institute for Advanced Architecture of Catalonia (IAAC). Worked example for **AI Methods I: Machine Learning for the Built Environment**, MaAI 2026/27.

The model and code are released under the **MIT License**. The training data is subject to its own licence, **CDLA-Sharing-1.0**; under that licence, results such as a trained model are not restricted by the data licence.
