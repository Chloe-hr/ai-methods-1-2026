---
license: mit
library_name: sklearn
tags:
  - sklearn
  - clustering
  - unsupervised-learning
  - kmeans
  - energy
  - load-profiles
  - built-environment
  - iaac-maai
  - ai-methods-1
datasets:
  - iaac-maai/day3-example-daily-load-profiles
---

# Household load profiles: grouping days by the shape of their electricity use

This model assigns **one day of household electricity use** (24 hourly values in kW) to one of **four typical daily profiles**: two kinds of workday, an "at home all day" profile and a "nobody home" profile. It is a scikit-learn pipeline (`Normalizer` + `KMeans`) fitted on 1,411 days measured in one house near Paris between 2006 and 2010.

It is the **worked example** for the Day 3 assignment of **AI Methods I: Machine Learning for the Built Environment** (IAAC MaAI 2026/27): it shows students what a published clustering model and its model card should look like.

- **GitHub repository (full analysis):** https://github.com/iaac-maai/ai-methods-1-2026 (notebook `Day_3/docs/application_of_unsupervised_learning.ipynb`, Application B and section 8)
- **Dataset with the cluster of every day:** https://huggingface.co/datasets/iaac-maai/day3-example-daily-load-profiles
- **Authors:** AI Methods I course team, IAAC

## Intended use

- **Intended users and use:** students, building managers and energy designers who want to **label days by their pattern of use**, for example to count how many "empty house" workdays a dwelling has (candidates for lowering the heating setpoint during the day) or to spot holiday periods in meter data. It is a **teaching model**.
- **Out-of-scope uses:**
  - Other building types (offices, schools, shops) or whole districts: their days look nothing like a household's, but the model will still force them into one of the four profiles.
  - Households with very different equipment (electric car charging at night, heat pumps, rooftop PV): their curves were not in the data.
  - Predicting **how much** electricity a day will use: the model only looks at the **shape** of the curve, never the total.
  - Any decision about a specific household or its occupants.

## Model details

| | |
|---|---|
| **Task** | Clustering (unsupervised) |
| **Algorithm** | `KMeans` |
| **Key hyperparameters** | `n_clusters=4`, `n_init=10`, `random_state=42` |
| **Preprocessing** | `Normalizer(norm="l1")`: each day is divided by its own total, so the model groups the **shape** of the curve (the share of the day's electricity used in each hour), not the volume. No `StandardScaler`: all 24 columns are in the same unit |
| **File** | `model.joblib`: the whole fitted scikit-learn `Pipeline` (8 KB) |
| **Library versions** | scikit-learn 1.9.1, pandas 3.0.6, numpy 2.5.3, Python 3.12.3 |

## Input features

The model expects a `pandas.DataFrame` with exactly these **24 columns**, one row per day:

| Column | Type | Unit | Description | Preprocessing |
|---|---|---|---|---|
| `h00` … `h23` | numeric | kW | Average active power of the whole household during hour 0 (00:00–00:59) … hour 23 | Each row divided by its sum (`Normalizer(norm="l1")`) |

All values must be present (no missing hours) and non-negative.

## Output and clusters

`model.predict(X)` returns the profile number of each day:

| Cluster | Name | Share of the days | Shape of the day | Calendar (never used by the model) |
|---|---|---|---|---|
| 0 | **workday, home in the morning** | 33.5% | Sharp peak at 7:00, a second peak around 20:00, moderate use in between | 90% weekdays; ~26 kWh/day |
| 1 | **at home all day** | 26.3% | Low at night, rising through the day, highest in the evening (peak 19:00) | 61% weekends, most common in winter (Nov–Jan); ~32 kWh/day |
| 2 | **workday, empty house** | 25.6% | Morning peak, almost nothing during the day, strong evening peak at 21:00 | 91% weekdays; ~24 kWh/day |
| 3 | **nobody home** | 14.7% | Flat: no peaks, only the base load (fridge, standby) | 55% of all August days, 28% of July; ~17 kWh/day |

The names come from our interpretation: we compared each cluster with information the model never saw (day of the week, month, daily total).

![The four profiles](load_profiles.png)

*Red: the centroid (average day) of each profile. Grey: 40 real days from it. Values are shares of the day's electricity.*

## Training data

| | |
|---|---|
| **Dataset** | Individual Household Electric Power Consumption (Hebrail & Berard), processed into daily profiles |
| **Source** | [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption) (also on [Kaggle](https://www.kaggle.com/datasets/uciml/electric-power-consumption-data-set)); our processed version: [iaac-maai/day3-example-daily-load-profiles](https://huggingface.co/datasets/iaac-maai/day3-example-daily-load-profiles) |
| **Licence** | CC BY 4.0 |
| **One row is** | One day of one household (24 hourly averages) |
| **Size used** | 1,411 complete days |
| **Coverage** | One house in Sceaux (near Paris, France), 17 December 2006 – 25 November 2010 |

**Processing:** about 2 million one-minute readings of `Global_active_power` → hourly averages (an hour counts as missing if fewer than 45 of its 60 minutes are valid) → one row per day → only days with all 24 hours valid are kept (1,411 days).

## Method and evaluation

- **Choice of k:** there is **no natural number of clusters**: the silhouette score is low for every k from 2 to 8 (0.13 at k = 2, 0.10 at k = 4) and the elbow curve has no clear bend. Daily habits form a **continuum**. We chose **k = 4** for interpretability: four profiles that can be named and acted upon.
- **Stability** (adjusted Rand index, ARI, against the final clusters; 1 = identical grouping):

| Check | Mean ARI | Min ARI |
|---|---|---|
| 10 different random seeds | 0.98 | 0.96 |
| 10 random subsamples of 80% of the days | 0.85 | 0.45 |
| Hierarchical clustering (ward), k = 4 | 0.34 | – |

- **Reading:** the profiles are **reproducible** for K-Means and fairly robust to losing data, but a different algorithm cuts the continuum differently. Treat them as a **useful summary of this household's days**, not as natural categories.
- **Why this model:** K-Means can assign **new** days (`.predict()`), its centroids are readable "average days", and it is fast.

## Limitations and biases

- **One household.** The profiles describe the habits of one family in one house, in one climate, between 2006 and 2010. Other households have other profiles.
- **Always one of four.** K-Means has no "I don't know": a day unlike any of the four (an office, a party, a broken fridge) is still assigned to the nearest profile.
- **Weak structure.** With a silhouette of about 0.1, many days sit between two profiles; a small change in the curve can move them to the neighbouring profile.
- **Shape only.** The volume of the day is deliberately ignored, so a cold winter day and a mild day with the same pattern get the same profile.
- **Names are interpretations.** "Nobody home" is our reading of a flat curve that coincides with the summer holidays; the data says nothing about who was actually at home.

## How to use

```python
import joblib
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

path = hf_hub_download(repo_id="iaac-maai/day3-example-load-profiles", filename="model.joblib")
model = joblib.load(path)

CLUSTER_NAMES = {
    0: "workday, home in the morning",
    1: "at home all day",
    2: "workday, empty house",
    3: "nobody home",
}

# One day: average kW for each hour 0..23
day = np.full(24, 0.3)                                   # base load
day[[6, 7]] = [1.2, 1.5]                                 # breakfast
day[[18, 19, 20, 21, 22]] = [1.5, 2.5, 3.0, 2.8, 1.5]    # evening
example = pd.DataFrame([day], columns=[f"h{h:02d}" for h in range(24)])

cluster = model.predict(example)[0]
print(cluster, CLUSTER_NAMES[cluster])                   # 2 workday, empty house
```

⚠️ `.joblib` files are Python pickles and can execute code when loaded: only load models from sources you trust. Use the library versions listed above; the model may not load with other versions.

## Reproducibility

The full analysis (download, processing, choice of k, interpretation, stability, export) is in the GitHub repository: https://github.com/iaac-maai/ai-methods-1-2026. Run `Day_3/docs/application_of_unsupervised_learning.ipynb` with the packages in the repository's `requirements.txt`.

## Authors and licence

AI Methods I course team. Worked example for **AI Methods I: Machine Learning for the Built Environment**, IAAC MaAI 2026/27.

Model released under the MIT License. It was fitted on data derived from *Individual Household Electric Power Consumption* by G. Hebrail and A. Berard, UCI Machine Learning Repository, https://doi.org/10.24432/C58K54, licensed under CC BY 4.0.
