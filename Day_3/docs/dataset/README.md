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

## What the clusters look like

![The average day of each profile](https://huggingface.co/datasets/iaac-maai/day3-example-daily-load-profiles/resolve/main/dataset_average_days.png)

The average day of each profile, in kW (computed from the `h00` … `h23` columns):

- **workday, home in the morning:** a sharp breakfast peak at 7:00 (about 2.0 kW), moderate use through the morning, a second peak around 20:00.
- **workday, empty house:** the same breakfast peak, then the **lowest daytime use** of all profiles (about 0.55 kW at 15:00, only the base load) and the **highest peak of all** at 21:00 (about 2.6 kW).
- **at home all day:** use rises steadily through the day and peaks in the evening (about 2.4 kW at 19:00): someone is at home cooking, heating and using appliances.
- **nobody home:** a flat line between 0.6 and 1.1 kW, with no morning or evening peak: only the fridge and devices on standby.

![The days on the UMAP map and the profiles per month](https://huggingface.co/datasets/iaac-maai/day3-example-daily-load-profiles/resolve/main/dataset_clusters_overview.png)

**Left: how the clustering behaves.** Each dot is one day, placed on a 2D map (`umap_x`, `umap_y`) so that days with similar curves are neighbours. The four profiles occupy **neighbouring regions of one continuous cloud**, with no empty space between them: the two workday profiles mix along their border, and only the "nobody home" days form a tight group of their own (top left). The clusters are a useful way of **cutting a continuum**, not separate islands; this is why the silhouette score is low (about 0.1) and why another algorithm would draw the borders differently.

**Right: the profiles through the year** (`month` and `cluster_name` columns). "Nobody home" makes up **55% of August** and 28% of July (summer holidays); "at home all day" is most common from **November to January** (34–41% of the days); the two workday profiles dominate the rest of the year.

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
