---
license: cc-by-4.0
pretty_name: Daily household load profiles (Sceaux, 2006–2010), with typical-day clusters
size_categories:
  - 1K<n<10K
tags:
  - built-environment
  - energy
  - electricity
  - load-profiles
  - clustering
  - iaac-maai
  - ai-methods-1
---

# Daily household load profiles (Sceaux, 2006–2010), with typical-day clusters

**1,411 days** of electricity use of one household in Sceaux (near Paris, France), between December 2006 and November 2010. Each row is **one day**: the average power in each of the 24 hours, the daily total, calendar information, and the **typical-day profile** the day belongs to ("workday, home in the morning", "at home all day", "workday, empty house", "nobody home"), found by clustering.

It turns a well-known minute-by-minute dataset into a **ready-to-use daily table** for studying load profiles, and it is the **worked example** dataset for the Day 3 assignment of **AI Methods I: Machine Learning for the Built Environment** (IAAC MaAI 2026/27).

- **GitHub repository (how it was made):** https://github.com/iaac-maai/ai-methods-1-2026 (notebook `Day_3/docs/application_of_unsupervised_learning.ipynb`, Application B and section 8)
- **Model fitted on it:** https://huggingface.co/iaac-maai/day3-example-load-profiles
- **Authors:** AI Methods I course team, IAAC

## Source and attribution

| | |
|---|---|
| **Original dataset** | [Individual Household Electric Power Consumption](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption) by Georges Hebrail and Alice Berard, UCI Machine Learning Repository, https://doi.org/10.24432/C58K54 |
| **Original licence** | CC BY 4.0 |
| **Original content** | About 2 million one-minute measurements (December 2006 – November 2010) of the household's active and reactive power, voltage, current and three sub-meters |

This dataset is a derived work, released under the same licence (CC BY 4.0) with attribution to the original authors.

## What we changed

- **Selection:** only `Global_active_power` (the total active power of the household, in kW) is used.
- **Aggregation:** one-minute readings → **hourly averages**. An hour with fewer than 45 valid minutes (out of 60) is treated as missing.
- **Cleaning:** missing values (`?` in the original file) are ignored in the averages; only days with **all 24 hours** valid are kept: **1,411 days**.
- **Feature engineering:** the daily energy use (`daily_kwh`) and calendar features (`weekday`, `weekend`, `month`, `season`).
- **Clustering:** each day's 24-hour curve is divided by its own total (to compare **shapes**, not volumes) and grouped with K-Means into 4 typical-day profiles (`cluster`, `cluster_name`). The 2D map coordinates come from UMAP (`umap_x`, `umap_y`). Details, choice of k and stability checks: see the [model card](https://huggingface.co/iaac-maai/day3-example-load-profiles).

## Columns

| Column | Type | Unit | Description |
|---|---|---|---|
| `date` | date | – | The day (YYYY-MM-DD) |
| `h00` … `h23` | float | kW | Average active power of the whole household during hour 0 (00:00–00:59) … hour 23 |
| `daily_kwh` | float | kWh | Total electricity used that day (sum of the 24 hourly averages) |
| `weekday` | text | – | Day of the week (Monday … Sunday) |
| `weekend` | bool | – | `True` on Saturday and Sunday |
| `month` | integer | – | Month (1–12) |
| `season` | text | – | Meteorological season: winter (Dec–Feb), spring, summer, autumn |
| `cluster` | integer | – | Typical-day profile assigned by the model (0–3) |
| `cluster_name` | text | – | Name of the profile (our interpretation): `workday, home in the morning` (33.5% of days), `at home all day` (26.3%), `workday, empty house` (25.6%), `nobody home` (14.7%) |
| `umap_x`, `umap_y` | float | – | Position of the day on a 2D UMAP map of the 24-hour shapes. Only useful for plotting: distances and cluster sizes on such maps are not meaningful |

## Limitations and biases

- **One household.** These are the habits of one family in one house near Paris; they do not represent other households, building types or climates.
- **Gaps.** Days with missing hours were removed, so some periods (including a few longer outages in the original data) are absent.
- **The clusters are one reading of the data.** Daily habits form a continuum (silhouette ≈ 0.1); another algorithm or another number of clusters would draw the borders differently. The names are our interpretation of the curves and the calendar, not information about the occupants.
- **The UMAP coordinates are not reproducible exactly** on another computer or library version.

## How to load it

```python
import pandas as pd

df = pd.read_csv(
    "hf://datasets/iaac-maai/day3-example-daily-load-profiles/daily_load_profiles_clustered.csv",
    parse_dates=["date"],
)
df.groupby("cluster_name")["daily_kwh"].median()
```

Reading `hf://` paths requires the `huggingface_hub` package.

## Authors and licence

AI Methods I course team. Worked example for **AI Methods I: Machine Learning for the Built Environment**, IAAC MaAI 2026/27.

Released under **CC BY 4.0**, based on *Individual Household Electric Power Consumption* by G. Hebrail and A. Berard (UCI Machine Learning Repository, https://doi.org/10.24432/C58K54, CC BY 4.0).
