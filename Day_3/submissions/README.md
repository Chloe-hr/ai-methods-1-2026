# DAY_3 SUBMISSIONS

## Assignment: an unsupervised learning analysis, published as a model (and a dataset)

Find and **explain** structure in built-environment data with clustering and dimensionality reduction, and **publish** the result.

You deliver:

1. A **GitHub repository** with a documented notebook that goes from data to **named, interpreted clusters**, structured like the Day 1 and Day 2 repositories.
2. The **fitted model on Hugging Face**, inside the course organization, with a **model card**, so anyone can assign a new example to one of your groups.
3. **Your dataset on Hugging Face, with a dataset card**, but **only if the dataset is your own work** (see [section 5.2](#52-the-dataset-publish-it-or-link-to-it)). Otherwise, a link to the original source is enough.

This assignment applies what we did in class ([`Day_3/docs/application_of_unsupervised_learning.ipynb`](../docs/application_of_unsupervised_learning.ipynb)) to your own data. The natural starting point is the **cleaned dataset from your Day 1 or Day 2 assignment**.

| | |
|---|---|
| **Work in** | Teams (as formed in class) |
| **Deliverables** | One GitHub repository in the [iaac-maai GitHub organization](https://github.com/iaac-maai), one model repository in the [iaac-maai Hugging Face organization](https://huggingface.co/iaac-maai) and, if eligible, one dataset repository there too |
| **Deadline** | **Monday, 2 November 2026, 23:59 CET** |
| **How to hand in** | See [section 7 of the course README](../../README.md#7-hand-in-your-submission), and put the Hugging Face link(s) in your GitHub README |

⚠️ **On 2 November 2026 the submission links (submodules) in this repository are updated to the latest commit on your `main` branch.** That commit, and the version of your Hugging Face repositories at that time, is your submission. Anything pushed afterwards is not part of it.

---

## Table of contents

1. [Choose a problem](#1-choose-a-problem)
2. [Repository structure](#2-repository-structure)
3. [The notebook](#3-the-notebook)
4. [The GitHub README](#4-the-github-readme)
5. [Publish on Hugging Face](#5-publish-on-hugging-face)
6. [The model card](#6-the-model-card)
7. [The dataset card](#7-the-dataset-card)
8. [The other files](#8-the-other-files)
9. [Submission checklist](#9-submission-checklist)
10. [Evaluation](#10-evaluation)

---

## 1. Choose a problem

There is no target today. Your question is: **which groups (or which main directions) exist in this data, and what do they mean for someone in the built environment?**

| Kind of question | Examples |
|---|---|
| **Typologies** | Building or dwelling types from geometry, age, materials; urban block types from morphology |
| **Profiles** | Typical days of energy use, occupancy patterns, traffic or sensor profiles |
| **Segmentation** | Housing markets, neighbourhoods, a building stock for a retrofit programme |
| **Outliers** | Buildings, meters or projects that behave unlike any group (DBSCAN noise) |
| **Spatial data** | Segmenting a point cloud into objects, grouping GPS traces |

Before you start, answer three questions. They shape every later decision and they go in your README and model card:

1. **Who would use these groups, and for what?** For example: "a municipality choosing which housing types to target first in a retrofit programme".
2. **Which features define "similar" for that question?** Choosing the columns is choosing the meaning of your clusters (remember Application A in class: same buildings, different features, completely different groups).
3. **What would make a cluster useful?** Usually: you can **name** it, and someone can **act** on it.

### Which dataset?

Use the **cleaned dataset from your Day 1 or Day 2 assignment** if it suits an unsupervised question. If it does not, you may switch to another built-environment dataset (see the [Day 1 brief](../../Day_1/submissions/README.md#1-choose-a-dataset) for where to look). Explain why in your README.

⚠️ **Switching datasets means two notebooks.** A new dataset has not been cleaned and explored yet, and you cannot interpret clusters of data you don't understand. If you use a dataset **different from your Day 1 and Day 2 submissions**, your Day 3 repository must contain **two notebooks**:

1. `notebook/data_cleaning_and_EDA.ipynb`: the **data cleaning and EDA** of the new dataset, following the [Day 1 brief](../../Day_1/submissions/README.md) (same required sections, the same leniency on cleaning, the same emphasis on EDA). It saves the cleaned dataset to `data/`.
2. `notebook/unsupervised_learning.ipynb`: the **unsupervised analysis** described in [section 3](#3-the-notebook), which loads the cleaned dataset produced by notebook 1.

Run them in that order. Your README must say so in *How to run*.

⚠️ **Do not use the three datasets from class** (house prices, energy efficiency and household power consumption). If you are unsure whether your problem is suitable, ask the course maintainers before you start.

💡 **Tip:** unsupervised methods need **enough rows** (at least a few hundred for clustering individual examples) and **several meaningful numeric features**. A dataset with one interesting column will not give interesting clusters.

---

## 2. Repository structure

Your repository must contain this structure:

```
<repo-name>/
├── data/
│   └── .gitkeep                       # keeps the folder in git
├── dataset/                           # ONLY if you publish your dataset (section 5.2)
│   ├── README.md                      # the Hugging Face dataset card (committed)
│   └── <dataset-name>.csv             # the published data (NOT committed: uploaded to Hugging Face)
├── figures/
│   ├── 01_pca_explained_variance.png  # your exported plots
│   └── ...
├── models/
│   ├── README.md                      # the Hugging Face model card (committed)
│   └── model.joblib                   # the fitted pipeline (NOT committed: uploaded to Hugging Face)
├── notebook/
│   ├── data_cleaning_and_EDA.ipynb    # ONLY if you switched to a new dataset (see section 1)
│   └── unsupervised_learning.ipynb
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

| Item | What goes in it |
|---|---|
| `data/` | Same rules as Day 1: raw data is downloaded by the notebook, not committed. A small cleaned dataset (a few MB) may be committed. |
| `dataset/` | Only if your dataset is eligible for publication: exactly what you upload to Hugging Face as a dataset. The dataset card (`README.md`) is committed; the data file lives **only** on Hugging Face. |
| `figures/` | The key plots, exported by the notebook as `.png` files, **committed** and shown in your README. |
| `models/` | Exactly what you upload to Hugging Face as a model: the model card (`README.md`, committed to GitHub) and the fitted pipeline (`model.joblib`, **only** on Hugging Face). |
| `notebook/` | Your unsupervised learning notebook. If you switched to a new dataset, **also** the data cleaning and EDA notebook for it. Other extra notebooks are allowed if you explain them in the README. |
| `.gitignore`, `LICENSE`, `requirements.txt` | See [section 8](#8-the-other-files). |
| `README.md` | The front page of your project. See [section 4](#4-the-github-readme). |

As on Day 2: your GitHub repository holds the **code that produces the results**; Hugging Face holds the **model and the data themselves**. Each one links to the other.

⚠️ **Paths:** the notebook lives in `notebook/`, so the other folders are **one level up**. Use relative paths only:

```python
from pathlib import Path

DATA_DIR = Path("..") / "data"
FIG_DIR = Path("..") / "figures"
MODEL_DIR = Path("..") / "models"
DATASET_DIR = Path("..") / "dataset"   # only if you publish your dataset
```

---

## 3. The notebook

The notebook must **run from top to bottom without errors** (*Restart kernel → Run All*) and must be committed **with its outputs**. As before, every step needs a **markdown cell that explains what you do and why**. Today there is no score that tells you whether you are right, so the decisions that matter most are **which features you use, how you choose the number of groups, and how you interpret them**.

### 3.1 Required sections

**0. Introduction and problem definition**
- Title, team members, and the **use case**: who uses the groups, for what, and what would make a group useful.
- Load the data: the cleaned Day 1/2 dataset or, if you switched datasets, the cleaned dataset saved by your `data_cleaning_and_EDA.ipynb`.
- State your **unit of analysis** (what one row is: a building, a day, a city, a point) and, if needed, how you aggregated the data to get it (e.g. minute readings → one row per day).

**1. Features and preprocessing**
- List the features you **cluster on**, and the ones you **keep aside for interpretation**. Justify the choice: it defines what "similar" means.
- **Scale** the features (`StandardScaler`, inside a `Pipeline`), unless they already share a unit (e.g. x, y, z in metres): say which, and why.
- Handle skewed columns (e.g. a log transform for prices or areas) and categorical columns (one-hot, or leave them out and use them for interpretation), and explain your choice.

**2. Dimensionality reduction**
- **PCA**: the explained variance plot and a **reading of the loadings**. What does each of the first components mean? How many real dimensions does your data have?
- **t-SNE or UMAP** (at least one) to draw the data in 2D. Comment on what it shows, keeping in mind that **cluster sizes and distances between groups in these plots are not meaningful**.

**3. Clustering**
- **At least two algorithms**: K-Means **and** hierarchical clustering or DBSCAN / HDBSCAN.
- **Choosing k (or eps)**: the elbow and silhouette plots for K-Means; the dendrogram for hierarchical; the k-distance plot for DBSCAN. If no k stands out (a common situation with real data), say so and explain how you chose.
- **Compare** the algorithms: do they agree (adjusted Rand index between them)? Which one fits the shape of your data better, and why?

**4. Stability**
- Show that your clusters are not an accident: re-run with different `random_state` values, or on a few random subsamples (e.g. 80% of the rows), and compare the results with the adjusted Rand index. Comment briefly.

**5. Interpretation (the most important section)**
- **Profile** every cluster: size, and average (or median) values of the features.
- **Cross-tabulate** the clusters with variables you did **not** cluster on (location, year, type, a performance measure…), as we did with the calendar in Application B and the heating load in Application A.
- **Name** every cluster ("compact pre-1960 blocks", "evening-peak weekdays"…) in a summary table. If you cannot name a cluster, say so and discuss why.
- Plot the clusters on the PCA and/or t-SNE/UMAP projection.

**6. Save and reuse the model**
- Save the **whole fitted pipeline** (scaler + optional PCA + clustering model) to `models/model.joblib` with `joblib.dump`.
- **Reload** it and assign a cluster to **one hand-made example** (e.g. a new building, a new day).
- K-Means has `.predict()`. **DBSCAN and agglomerative clustering do not**: they cannot assign new examples. If your final model is one of those, train a small **`KNeighborsClassifier`** on your data with the cluster labels as the target (inside the same pipeline, with the scaler), and save that as the "assigner". Explain this choice in the notebook and in the model card.

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# labels = the cluster of each row found by DBSCAN / agglomerative clustering
assigner = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))
assigner.fit(X, labels)            # X = the same feature columns you clustered on
assigner.predict(new_example)      # -> the cluster of a new example
```

**7. Conclusions**
- The **key findings** (3–6 bullet points, with numbers): which groups exist, how big they are, what distinguishes them.
- **Limitations and biases**: data, choices (features, k), stability, and what the clusters should **not** be used for.
- If you publish your dataset: export the final table (cleaned data + engineered features + cluster labels, see [section 5.3](#53-publish-your-dataset-only-if-eligible)) to `dataset/`.

### 3.2 Technical requirements

- Libraries: **pandas**, **numpy**, **matplotlib**/**seaborn** and **scikit-learn**; **umap-learn** if you use UMAP; **scipy** for dendrograms.
- Scaling and the model inside a **`Pipeline`**.
- A fixed `random_state` everywhere randomness is involved (K-Means, t-SNE, UMAP, subsampling).
- At least **4 figures** exported to `figures/` (e.g. PCA explained variance, choice of k, clusters in 2D, cluster profiles).
- **No credentials** in the notebook: no Kaggle keys, **no Hugging Face tokens**.

---

## 4. The GitHub README

Someone who has never seen your project should understand it and be able to run it **in under five minutes**. It must contain:

1. **Title** and **team members**.
2. **Use case**: who uses the groups, for what (2–4 sentences).
3. **On Hugging Face**: the link to your model repository and, if published, your dataset repository, at the top, impossible to miss.
4. **Dataset**: name, link (to the original source **and** to your Hugging Face dataset if published), short description, licence.
5. **Repository structure**: the folder tree with one line per item.
6. **How to run**: environment, `requirements.txt`, data, notebook(s) and their order.
7. **Approach**: unit of analysis, features used (and kept aside), preprocessing, methods compared.
8. **Results**: the **table of named clusters** (name, size, defining features) and **at least 2 figures** embedded from `figures/`.
9. **Limitations**.
10. **AI usage disclosure**: same rules as Days 1 and 2, see [below](#ai-usage-disclosure).
11. **License**.

### AI usage disclosure

Using AI tools is allowed, but it must be disclosed. As before, this section is also a way to **share with the class how to use agentic tools effectively**, so be concrete:

- **Which tools** did you use (e.g. ChatGPT, Claude, Claude Code, GitHub Copilot, Cursor)? Did **agents** edit files or run code by themselves?
- **Why**: what did you expect the tool to help with?
- **Where**: which parts (feature engineering, clustering code, plots, interpretation, model and dataset cards, README)?
- **How**: your workflow. What did you ask for (an example prompt helps), how did you check the output, and what did you correct or reject?
- **Lessons learned**: what worked, what did not, what would you do differently?

If you did not use AI tools, write one sentence saying so.

⚠️ You are responsible for everything in your submission, including what an AI tool produced. **The names and interpretations of your clusters must be your team's own reading of the results**, and you must be able to defend them.

A starting template:

````markdown
# <Project title>

**Team:** Name Surname, Name Surname, Name Surname
**Course:** AI Methods I – Day 3: Unsupervised learning (IAAC MaAI 2026/27)
**Model on Hugging Face:** https://huggingface.co/iaac-maai/<model-repo>
**Dataset on Hugging Face:** https://huggingface.co/datasets/iaac-maai/<dataset-repo>  <!-- or: "not published, see source below" -->

## Use case
...

## Dataset
| | |
|---|---|
| **Name** | ... |
| **Source** | [link](...) |
| **Rows used** | ... (one row = ...) |
| **Licence** | ... |

## Repository structure
```
├── data/        # downloaded and cleaned data (not committed)
├── dataset/     # dataset card (README.md); the data lives on Hugging Face   <!-- only if published -->
├── figures/     # exported plots
├── models/      # model card (README.md); model.joblib lives on Hugging Face
├── notebook/    # unsupervised_learning.ipynb (+ data_cleaning_and_EDA.ipynb if new dataset)
...
```

## How to run
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
Open `notebook/unsupervised_learning.ipynb`, select the `.venv` kernel and run all cells.
(If you switched datasets: run `notebook/data_cleaning_and_EDA.ipynb` first. It creates the cleaned dataset that the second notebook loads.)

## Approach
- **Unit of analysis:** one row = ...
- **Features clustered on:** ... (kept aside for interpretation: ...)
- **Preprocessing:** ...
- **Methods:** PCA, UMAP / t-SNE, K-Means, ... ; final choice: ... because ...

## Results
| Cluster | Name | Size | Defining features |
|---|---|---|---|
| 0 | ... | ... | ... |
| 1 | ... | ... | ... |

![Clusters in 2D](figures/03_clusters_umap.png)

## Limitations
- ...

## AI usage disclosure
| | |
|---|---|
| **Tools** | e.g. Claude Code (agent), ChatGPT (chat) / none |
| **Why** | ... |
| **Where** | ... |
| **How** | ... |

**Lessons learned:** ...

## License
Code released under the [MIT License](LICENSE). The dataset is subject to its own licence (see above).
````

---

## 5. Publish on Hugging Face

You publish exactly as on Day 2. If you have not set up Hugging Face yet (account, membership of the **iaac-maai** organization, the `hf` command-line tool, an access token and `hf auth login`), follow **[section 5.1 of the Day 2 brief](../../Day_2/submissions/README.md#51-one-time-setup)** first.

⚠️ **Never put the token in your notebook, your code or your repository.**

### 5.1 The model (always)

**Name** your model repository with this pattern, all lowercase, with hyphens:

```
iaac-maai/day3-<team>-<what-it-groups>
```

For example `iaac-maai/day3-team3-dwelling-typologies` or `iaac-maai/day3-team5-load-profiles`.

**Create and upload** from the **root of your GitHub repository**, after the notebook has saved `models/model.joblib` and you have written `models/README.md` (the model card, [section 6](#6-the-model-card)):

```bash
# 1. Create the model repository inside the organization
hf repos create iaac-maai/day3-team3-dwelling-typologies --type model --exist-ok

# 2. Upload the contents of models/ (model.joblib + README.md) to the root of the repository
hf upload iaac-maai/day3-team3-dwelling-typologies models . --commit-message "Upload fitted pipeline and model card"
```

**Test it like a stranger**: download it into a fresh folder and assign a cluster to one example:

```bash
hf download iaac-maai/day3-team3-dwelling-typologies --local-dir test_download
```

```python
import joblib
import pandas as pd

model = joblib.load("test_download/model.joblib")
example = pd.DataFrame([{ ... }])   # one row with the same columns you clustered on
print(model.predict(example))       # -> the cluster number; your model card says what it means
```

If this fails (missing package, wrong scikit-learn version, missing column), fix it before the deadline. Delete `test_download/` afterwards; do not commit it.

### 5.2 The dataset: publish it, or link to it?

Publishing a dataset is a statement that **it is yours to share and worth sharing**. Re-uploading someone else's dataset with a few renamed columns is neither: it duplicates the original and hides who made it. Use this table:

| Your situation | What to do |
|---|---|
| You **collected or created** the data yourselves (survey, your own scraping, sensors, a simulation you ran, digitised drawings or plans) | **Publish** it on Hugging Face |
| You **transformed** an existing dataset **substantially**: merged several sources, **heavy cleaning**, substantial **imputation**, **feature engineering** that creates new information (e.g. daily profiles from minute readings, per-building indicators from raw records, geometric features from footprints) | **Publish** it on Hugging Face, with the original as source and attribution |
| **Light cleaning** of a downloaded dataset (renamed columns, dropped rows or duplicates, fixed types, a few imputed values) | **Do not publish.** Link to the original dataset in your README and model card |
| The original **licence does not allow redistribution** or derivatives (e.g. *NoDerivatives*, "for research use only", no licence stated) | **Do not publish**, even if your work is transformative. Link to the original |
| The data contains **personal information** (names, addresses, identifiable households) | **Never publish** |

⚠️ **Adding a cluster column is not enough to make a dataset new.** The transformation must be real work on the data itself. When in doubt, ask the course maintainers before uploading.

💡 **Tip:** open licences such as **CC BY 4.0**, **CC BY-SA 4.0**, **ODbL** and **CDLA** allow publishing derived datasets **with attribution**; *ShareAlike* licences (BY-SA, ODbL, CDLA-Sharing) also require you to publish your version **under the same licence**. Write the original licence in your dataset card.

### 5.3 Publish your dataset (only if eligible)

**What to publish:** one table, the final result of your notebook. It contains:
- the cleaned data and your **engineered features**;
- your **cluster labels** (`cluster`, and `cluster_name` with the names from your interpretation);
- optionally the 2D coordinates you plotted (`pca_1`, `pca_2` or `umap_x`, `umap_y`).

Export it from the notebook as **CSV** (e.g. `dataset/<dataset-name>.csv`). Keep it reasonably small (a few hundred MB at most) and check once more that it contains **no personal data**.

**Name** your dataset repository:

```
iaac-maai/day3-<team>-<dataset-name>
```

**Create and upload** (note `--type dataset` in **both** commands; without it, the files go to a *model* repository):

```bash
# 1. Create the dataset repository inside the organization
hf repos create iaac-maai/day3-team3-barcelona-dwellings --type dataset --exist-ok

# 2. Upload the contents of dataset/ (the data file + README.md) to the root of the repository
hf upload iaac-maai/day3-team3-barcelona-dwellings dataset . --type dataset --commit-message "Upload dataset and dataset card"
```

The dataset page is at `https://huggingface.co/datasets/iaac-maai/<your-repo>` (note `/datasets/` in the address). Check that the **Dataset Viewer** shows your table and that the card renders correctly.

**Test it like a stranger**: anyone with `pandas` and `huggingface_hub` installed can read it directly:

```python
import pandas as pd

df = pd.read_csv("hf://datasets/iaac-maai/day3-team3-barcelona-dwellings/barcelona_dwellings.csv")
df.head()
```

---

## 6. The model card

The model card is the `README.md` of your Hugging Face model repository (written in `models/README.md`, committed to GitHub, uploaded to Hugging Face). As on Day 2, it is the **user manual of your model**. For a clustering model, the most important part is the **meaning of each cluster**: a model that returns "cluster 3" is useless unless the card says what cluster 3 is.

💡 **Tip:** the class folder [`Day_3/docs/`](../docs/) is organised exactly like your submission and contains a complete worked example: the fitted load-profile model with its model card ([`docs/models/README.md`](../docs/models/README.md)) and the published dataset with its dataset card ([`docs/dataset/README.md`](../docs/dataset/README.md)). Section 8 of the class notebook shows how they were produced.

### 6.1 What it must contain

| Section | What to write |
|---|---|
| **Metadata header** | Licence, library (`sklearn`), tags (including `clustering` / `unsupervised-learning`), and `datasets:` if you published your dataset |
| **Model summary** | One paragraph: what it groups, from what, for whom, with which algorithm |
| **Intended use** | Who should use it and for what. **Out-of-scope uses**: where it must *not* be used (other cities, building types, value ranges…) |
| **Model details** | Algorithm and key **hyperparameters** (k, eps / min_samples, linkage), preprocessing (scaling, PCA), the "assigner" if you used one (section 3.1, step 6), file format, library versions, authors |
| **Input features** | A table: every input column, its type, unit, description and preprocessing. The **exact column names** the model expects |
| **Output and clusters** | What `.predict()` returns, and a **table of the clusters**: number, **name**, size (share of the data), defining features. Mention noise (`-1`) if relevant |
| **Training data** | Dataset name, **source link** (and your Hugging Face dataset if published), licence, size, what one row represents, coverage, a summary of the cleaning |
| **Method and evaluation** | Algorithms compared; how k / eps was chosen (silhouette, elbow, dendrogram…); the agreement between methods and the **stability** check; why this model was chosen |
| **Limitations and biases** | What the data does not represent, how sensitive the groups are to your choices, weak or unnamed clusters. Be honest: this is the most important section for a user |
| **How to use** | A short, working Python snippet: download, load, assign a cluster to one example, and translate the number into the cluster name |
| **Reproducibility** | Link to the **GitHub repository**, the notebook, and the `requirements.txt` / library versions |
| **Security note** | `.joblib` files are pickles: they can run code when loaded. Users should only load models from sources they trust |
| **Authors, course and licence** | Team members, "AI Methods I, IAAC MaAI 2026/27", licence of the model and of the data |

### 6.2 Template

Copy this into `models/README.md` and fill in every `...`. Delete the comments (`<!-- -->`) when you are done.

````markdown
---
license: mit
library_name: sklearn
tags:
  - sklearn
  - clustering
  - unsupervised-learning
  - built-environment
  - iaac-maai
  - ai-methods-1
datasets:                               # only if you published your dataset; otherwise delete these two lines
  - iaac-maai/<dataset-repo>
---

# <Model name>: grouping <what> by <which features>

<!-- One paragraph: what it groups, from what, for whom, with which algorithm. -->
This model assigns ... to one of <k> groups based on ... It is a scikit-learn pipeline (scaling + <algorithm>) fitted on ... It was built as a course assignment for AI Methods I (IAAC MaAI 2026/27).

- **GitHub repository (full analysis):** https://github.com/iaac-maai/<repo-name>
- **Team:** Name Surname, Name Surname, Name Surname

## Intended use

- **Intended users and use:** ...
- **Out-of-scope uses:** ... <!-- e.g. other cities, building types or value ranges not in the data; any use as a prediction of performance -->

## Model details

| | |
|---|---|
| **Task** | Clustering (unsupervised) |
| **Algorithm** | e.g. KMeans |
| **Key hyperparameters** | e.g. n_clusters=4, n_init=10, random_state=42 |
| **Preprocessing** | e.g. log10 of area, StandardScaler |
| **Assigner** | <!-- only for DBSCAN / agglomerative: e.g. KNeighborsClassifier(n_neighbors=5) trained on the cluster labels --> |
| **File** | `model.joblib`: the whole fitted scikit-learn `Pipeline` |
| **Library versions** | scikit-learn ..., pandas ..., numpy ..., Python ... |

## Input features

The model expects a `pandas.DataFrame` with exactly these columns:

| Column | Type | Unit | Description | Preprocessing |
|---|---|---|---|---|
| ... | numeric | m² | ... | log10, scaling |
| ... | numeric | – | ... | scaling |

## Output and clusters

`model.predict(X)` returns the cluster number of each row:

| Cluster | Name | Share of the data | Defining features |
|---|---|---|---|
| 0 | ... | ...% | ... |
| 1 | ... | ...% | ... |

## Training data

| | |
|---|---|
| **Dataset** | ... |
| **Source** | [link](...) <!-- + the Hugging Face dataset if published --> |
| **Licence** | ... |
| **One row is** | ... |
| **Size used** | ... rows |
| **Coverage** | location, period, building types... |

**Cleaning and feature engineering:** ...

## Method and evaluation

- **Methods compared:** ... (agreement between them: ARI = ...)
- **Choice of k / eps:** ... (silhouette = ..., elbow at ..., dendrogram ...)
- **Stability:** ... (mean ARI over ... runs / subsamples = ...)
- **Why this model:** ...

## Limitations and biases

- ...

## How to use

```python
import joblib
import pandas as pd
from huggingface_hub import hf_hub_download

path = hf_hub_download(repo_id="iaac-maai/<model-repo>", filename="model.joblib")
model = joblib.load(path)

CLUSTER_NAMES = {0: "...", 1: "..."}

example = pd.DataFrame([{
    "column_1": ...,
    "column_2": ...,
}])
cluster = model.predict(example)[0]
print(cluster, CLUSTER_NAMES[cluster])
```

⚠️ `.joblib` files are Python pickles and can execute code when loaded: only load models from sources you trust. Use the library versions listed above; the model may not load with other versions.

## Reproducibility

The full analysis (data loading, cleaning, feature engineering, clustering, interpretation) is in the GitHub repository: https://github.com/iaac-maai/<repo-name>. Run `notebook/unsupervised_learning.ipynb` with the packages in `requirements.txt`.

## Authors and licence

Name Surname, Name Surname, Name Surname. Course assignment for **AI Methods I: Machine Learning for the Built Environment**, IAAC MaAI 2026/27.

Model released under the MIT License. The data is subject to its own licence: ...
````

💡 **Tip:** to show a figure in the model card (e.g. the cluster profiles), copy it into `models/`, embed it with `![Cluster profiles](cluster_profiles.png)` and upload again.

---

## 7. The dataset card

**Only if you publish your dataset.** The dataset card is the `README.md` of your Hugging Face dataset repository (written in `dataset/README.md`, committed to GitHub, uploaded to Hugging Face). It tells a reader **where the data comes from, what you did to it, and what every column means**, so they can reuse it without opening your notebook.

### 7.1 What it must contain

| Section | What to write |
|---|---|
| **Metadata header** | Licence (compatible with the original, see [5.2](#52-the-dataset-publish-it-or-link-to-it)), a readable name, size category, tags |
| **Summary** | One paragraph: what the data describes, where, when, one row = ? |
| **Source and attribution** | The **original dataset(s)**: name, link, authors, licence. Or how **you collected** the data |
| **What we changed** | The cleaning, imputation and **feature engineering** you did, with numbers (rows removed, values imputed, features created). This paragraph shows why your version is **new** |
| **Columns** | A table: every column, its type, unit and meaning, including the engineered features and `cluster` / `cluster_name` |
| **Limitations and biases** | What the data does not represent, sampling bias, imputation effects, the clusters being one possible reading |
| **How to load it** | A working snippet |
| **Related** | Links to the GitHub repository and the model on Hugging Face |
| **Authors and licence** | Team members, course, licence |

### 7.2 Template

Copy this into `dataset/README.md` and fill in every `...`. For a filled-in example, see [`Day_3/docs/dataset/README.md`](../docs/dataset/README.md).

````markdown
---
license: cc-by-4.0                      # must be compatible with the original licence
pretty_name: <Readable dataset name>
size_categories:
  - 1K<n<10K                            # n<1K, 1K<n<10K, 10K<n<100K, 100K<n<1M ...
tags:
  - built-environment
  - clustering
  - iaac-maai
  - ai-methods-1
---

# <Readable dataset name>

<!-- One paragraph: what the data describes, where, when, and what one row is. -->
...

- **GitHub repository (how it was made):** https://github.com/iaac-maai/<repo-name>
- **Model fitted on it:** https://huggingface.co/iaac-maai/<model-repo>
- **Team:** Name Surname, Name Surname, Name Surname

## Source and attribution

| | |
|---|---|
| **Original dataset** | [name](link) by ... <!-- or: collected by the team, how and when --> |
| **Original licence** | ... |

## What we changed

- **Cleaning:** ...
- **Imputation:** ...
- **Feature engineering:** ...
- **Clustering:** the `cluster` and `cluster_name` columns come from the model above.

## Columns

| Column | Type | Unit | Description |
|---|---|---|---|
| ... | numeric | m² | ... |
| `cluster` | integer | – | Cluster number assigned by the model |
| `cluster_name` | text | – | Name of the cluster (our interpretation) |

## Limitations and biases

- ...

## How to load it

```python
import pandas as pd

df = pd.read_csv("hf://datasets/iaac-maai/<dataset-repo>/<file>.csv")
```

## Authors and licence

Name Surname, Name Surname, Name Surname. Course assignment for **AI Methods I: Machine Learning for the Built Environment**, IAAC MaAI 2026/27.

Released under ... <!-- the licence in the header -->, based on ... <!-- original dataset and its licence -->.
````

---

## 8. The other files

### `.gitignore`

The Day 2 version (see the [Day 2 brief](../../Day_2/submissions/README.md#7-the-other-files)), plus the published data file:

```gitignore
# virtual environments
.venv/
venv/

# Jupyter
.ipynb_checkpoints/

# data: downloaded by the notebook, not committed
data/*
!data/.gitkeep

# fitted model and published data: on Hugging Face, not on GitHub
models/*.joblib
dataset/*.csv
test_download/

# OS files
.DS_Store
Thumbs.db

# secrets
.env
kaggle.json
```

⚠️ Do **not** ignore `figures/`, `*.png`, `models/README.md` or `dataset/README.md`: they must be on GitHub.

### `requirements.txt`

Only the packages your notebook imports, **with versions**. This assignment adds scipy (dendrograms) and, if you use it, umap-learn, for example:

```text
pandas==3.0.6
numpy==2.5.3
matplotlib==3.11.2
seaborn==0.13.2
scikit-learn==1.9.1
scipy==1.18.1
umap-learn==0.5.12
joblib==1.6.0
huggingface_hub==2.2.0
kagglehub==1.0.2
```

Pin **scikit-learn** exactly: a saved model may not load with another version, and your model card refers to these versions.

### `LICENSE`

**MIT License** for your code and model, as before. The data keeps **its own licence**: mention it in the GitHub README, the model card and, if published, the dataset card (whose licence must be compatible with the original).

---

## 9. Submission checklist

**GitHub repository**
- [ ] The repository is inside the **iaac-maai** GitHub organization
- [ ] It has `data/`, `figures/`, `models/` (with the model card), `notebook/`, `.gitignore`, `LICENSE`, `README.md` and `requirements.txt`, plus `dataset/` (with the dataset card) if you publish your dataset
- [ ] No virtual environment, raw data, `.joblib` model file, published data file or credentials are committed
- [ ] The final version is pushed to `main` **before 2 November 2026**

**Notebook**
- [ ] If you use a dataset different from Days 1 and 2: a second notebook, `data_cleaning_and_EDA.ipynb`, covers the cleaning and EDA of the new dataset following the Day 1 brief, and the README says to run it first
- [ ] It runs from top to bottom without errors and is committed **with outputs**
- [ ] It states the use case and the unit of analysis, and justifies the features clustered on (and those kept aside)
- [ ] Scaling (and any transformation) is in a `Pipeline` and justified
- [ ] It includes **PCA** (explained variance + loadings interpretation) and **t-SNE or UMAP**
- [ ] It compares **at least two clustering algorithms**, with the choice of k / eps justified (elbow, silhouette, dendrogram or k-distance plot)
- [ ] It checks the **stability** of the clusters
- [ ] It **profiles and names** every cluster, using variables that were **not** clustered on
- [ ] It saves the pipeline to `models/model.joblib`, reloads it and assigns a cluster to a hand-made example
- [ ] It exports at least 4 figures to `figures/`

**Hugging Face model**
- [ ] The model repository is inside the **iaac-maai** Hugging Face organization and named `day3-<team>-<what-it-groups>`
- [ ] It contains `model.joblib` and a complete **model card** (every section of [6.1](#61-what-it-must-contain)), including the **table of named clusters**
- [ ] The model card links to the GitHub repository, and the GitHub README links to the model
- [ ] It loads and assigns a cluster in a fresh folder ([section 5.1](#51-the-model-always))

**Dataset**
- [ ] **Either** the dataset is eligible ([section 5.2](#52-the-dataset-publish-it-or-link-to-it)) and published as `iaac-maai/day3-<team>-<dataset-name>`, with a complete **dataset card** (every section of [7.1](#71-what-it-must-contain)), a compatible licence, attribution and no personal data
- [ ] **Or** it is not published, and the README and model card link to the original source

**README**
- [ ] Team, use case, Hugging Face link(s), dataset, structure, how to run, approach, results (named clusters + at least 2 figures), limitations, AI usage disclosure, licence

---

## 10. Evaluation

The assignment is graded following the course evaluation criteria (see the syllabus). In particular:

| Criterion | What we look for |
|---|---|
| **Completeness** | All required files, sections and steps are present; the notebook runs; the model is published and loads; the dataset is published (if eligible) or properly linked |
| **Interpretation** (main weight) | Clusters that are profiled, **named** and connected to variables that were not used to build them; a clear answer to the use-case question |
| **Methodology** | Features chosen on purpose, scaling, at least two algorithms compared fairly, a justified choice of k / eps, a stability check, dimensionality reduction used and read correctly |
| **Critical thinking** | Honesty about weak structure (a low silhouette, unstable or unnamed clusters); t-SNE / UMAP plots not over-read; limitations and biases; correct reasoning about whether the dataset may be published |
| **Communication** | Clear notebook, README, **model card** and dataset card that a stranger can follow; readable, labelled figures; an honest and useful AI usage disclosure |
| **Relevance** | The question and the groups are meaningful for architecture and the built environment |
| **Repository and publication conventions** | Clean structure, sensible `.gitignore`, pinned `requirements.txt`, licences, correct Hugging Face naming, metadata and repository types |

A modest silhouette with a well-argued interpretation is worth more than a perfect score that nobody can explain.
