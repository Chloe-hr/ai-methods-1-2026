# DAY_1 SUBMISSIONS

## Assignment: data cleaning and EDA pipeline

Build a **data cleaning and exploratory data analysis (EDA) pipeline** for a dataset related to the **built environment**, and publish it as a documented GitHub repository.

The **main focus of this assignment is the EDA**: understanding your dataset and answering questions about the built environment with statistics and visualizations. How much cleaning you need depends on your dataset. If it is already clean, a short, well-documented quality check is enough.

This assignment applies what we did in class with the house prices dataset ([`Day_1/docs/data_cleaning_and_EDA.ipynb`](../docs/data_cleaning_and_EDA.ipynb)) to a dataset of your choice. The cleaned dataset you produce should be **ready for modelling**: you may reuse it in the Day 2 assignment.

| | |
|---|---|
| **Work in** | Teams (as formed in class) |
| **Deliverable** | One GitHub repository inside the [iaac-maai](https://github.com/iaac-maai) organization |
| **Deadline** | **Monday, 19 October 2026, 23:59 CET** |
| **How to hand in** | See [section 7 of the course README](../../README.md#7-hand-in-your-submission) |

⚠️ **On 19 October 2026 the submission links (submodules) in this repository are updated to the latest commit on your `main` branch.** That commit is your submission: anything pushed afterwards is not part of it.

---

## Table of contents

1. [Choose a dataset](#1-choose-a-dataset)
2. [Repository structure](#2-repository-structure)
3. [The notebook](#3-the-notebook)
4. [The README](#4-the-readme)
5. [The other files](#5-the-other-files)
6. [Submission checklist](#6-submission-checklist)
7. [Evaluation](#7-evaluation)

---

## 1. Choose a dataset

Pick a **tabular dataset** (a table: rows and columns) related to architecture, construction, real estate, energy or cities. Some places to look:

| Source | Examples |
|---|---|
| [Kaggle](https://www.kaggle.com/datasets) | Real estate listings, building energy use, construction costs |
| City open data portals | [Open Data BCN](https://opendata-ajuntament.barcelona.cat/en), [NYC Open Data](https://opendata.cityofnewyork.us) (building permits, energy benchmarking, cadastre) |
| National / EU statistics | [Eurostat](https://ec.europa.eu/eurostat/data/database), energy performance certificate registries |
| [UCI ML Repository](https://archive.ics.uci.edu) / [Hugging Face](https://huggingface.co/datasets) | Building energy, materials, sensor data |
| Your own work | BIM schedules exported to CSV, sensor logs, survey data |

A good dataset for this assignment:

- has **at least ~1,000 rows** and **several columns** of different types (numbers and categories);
- has **interesting variables to explore**: enough to ask and answer real questions about buildings or cities;
- has a **variable you could predict later** (a price, an energy demand, a class), so it can be reused on Day 2;
- has a **licence that allows reuse**. Check it, and write it in your README.

A **clean dataset is perfectly fine.** Finding a "dirty" dataset is not a requirement: many good open datasets are already curated. A messy dataset gives you more to show in the cleaning part, but the grade does not depend on how dirty your data is.

⚠️ **Do not use the house prices dataset from class.** If you are unsure whether your dataset is suitable, ask the course maintainers before you start.

---

## 2. Repository structure

Your repository must contain exactly this structure:

```
<repo-name>/
├── data/
│   └── .gitkeep                   # keeps the folder in git (see below)
├── figures/
│   ├── 01_missing_values.png      # your exported plots
│   └── ...
├── notebook/
│   └── data_cleaning_and_EDA.ipynb
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

| Item | What goes in it |
|---|---|
| `data/` | The data the notebook reads and writes. **Raw datasets are not committed**: the notebook downloads them (e.g. with `kagglehub`) and saves them here. See the note below. |
| `figures/` | The key plots, exported by the notebook as `.png` files. **These are committed**: they are shown in your README. |
| `notebook/` | Your notebook (one main notebook; extra notebooks are allowed if you explain them in the README). |
| `.gitignore` | Files git must not track. See [section 5](#5-the-other-files). |
| `LICENSE` | The licence of **your code**. See [section 5](#5-the-other-files). |
| `README.md` | The front page of your project. See [section 4](#4-the-readme). |
| `requirements.txt` | The Python packages (and versions) needed to run the notebook. |

**About `data/`:**

- If your dataset **can be downloaded** by code (Kaggle, a URL, an API), do not commit it. The notebook downloads it, and `data/` only contains `.gitkeep`.
- If it **cannot be downloaded** automatically (e.g. your own BIM export), you may commit it **only if it is smaller than 50 MB** and its licence allows it. Explain this in the README. GitHub rejects files above 100 MB.
- The **cleaned** dataset produced by your notebook can be committed if it is small (a few MB), so the Day 2 assignment can start from it.

⚠️ **Paths:** the notebook lives in `notebook/`, so the other folders are **one level up**. Use relative paths like this, never absolute paths such as `C:/Users/...` or `/home/...`:

```python
from pathlib import Path

DATA_DIR = Path("..") / "data"
FIG_DIR = Path("..") / "figures"
```

---

## 3. The notebook

The notebook is the core of the assignment. It must **run from top to bottom without errors** (*Restart kernel → Run All*) and must be committed **with its outputs**, so the results are visible on GitHub.

Use the class notebook as a model. Every step needs a **markdown cell that explains what you do and why**, and code with comments. **The decisions matter more than the code:** *why* did you drop this column, *why* did you fill these values with the median, *why* is this value an outlier and not a genuine extreme?

### 3.1 Required sections

**0. Introduction and data acquisition**
- Title, team members, and the **use case**: what question about the built environment does this data help answer?
- Load the data **from the notebook** (download it, or read it from `data/`).
- A **dataset card**: source (link), origin, what one row represents, size, licence, and known limitations.
- A first look: `shape`, `head()`, `info()`, `describe()`, and a short **data dictionary** (what each column means).

**1. Data quality check and cleaning.** The amount of cleaning depends on your dataset; it can be short. What is required is that you **check** the data and **report** what you found:
- **Check** for the classic problems: **duplicates**, **missing values**, **inconsistent data types** (numbers stored as text, mixed units, inconsistent spelling of categories) and **outliers**.
- If you find problems, **fix them and justify each decision** (e.g. why you dropped a column, why you filled values with the median, why a value is an error and not a genuine extreme).
- If the dataset is already clean, **show the checks** (e.g. `df.duplicated().sum()`, a missing-values table, a boxplot) and state in one or two sentences that no cleaning was needed. That is a valid result.
- **Recommended:** at least **one new, meaningful column** created from the existing ones (*feature engineering*), ideally with an architectural meaning, like the net-to-gross efficiency ratio in class.
- Save the dataset you will use for the analysis (cleaned or not) to `data/`.

**2. Exploratory Data Analysis (the core of the assignment)**
- **Descriptive statistics** of the main variables, explained in plain language.
- **Univariate** analysis: distributions of numeric variables and counts of categorical variables.
- **Bivariate** analysis: relations between pairs of variables, especially with the variable you might predict later.
- **Grouped comparisons**: `groupby` tables or plots comparing categories (e.g. by city, building type, construction period).
- **Correlation** analysis (e.g. a heatmap), with a comment on correlation vs causation.
- At least **three questions about the built environment**, each stated clearly, answered with a plot, and followed by a short written interpretation (what you see, what it might mean, and its limits).

**3. Conclusions**
- The **key findings** (3–6 bullet points, with numbers).
- **Limitations and biases** of the data (and of your cleaning decisions, if any).
- A short **"ready for modelling"** statement: what the target could be, and which columns must *not* be used as inputs (leakage).

### 3.2 Technical requirements

- Libraries: **pandas**, **numpy**, **matplotlib** and **seaborn** (scikit-learn is allowed, e.g. for imputation).
- Every plot has a **title**, **axis labels with units**, and a readable size.
- At least **4 key figures** exported to `figures/` (e.g. with a `save_fig()` helper as in class).
- A fixed random seed (e.g. `random_state=42`) for anything random, such as sampling.
- **No credentials** in the notebook: no Kaggle keys, tokens or passwords.

---

## 4. The README

The README is the first thing anyone sees. Someone who has never seen your project should understand it and be able to run it **in under five minutes**. It must contain:

1. **Title** and **team members**.
2. **Use case**: the problem or question, in 2–4 sentences.
3. **Dataset**: name, link, short description, licence (a compact version of your dataset card).
4. **Repository structure**: the folder tree with one line per item.
5. **How to run**: how to create the environment, install `requirements.txt`, get the data, and open the notebook.
6. **Data quality summary**: the problems you checked for, what you found, and how you solved them (a short table works well; "none found" is a valid answer).
7. **Key findings**: 3–6 bullet points, with **at least 2 figures** embedded from `figures/`.
8. **Limitations**.
9. **License**.

A starting template:

````markdown
# <Project title>

**Team:** Name Surname, Name Surname, Name Surname
**Course:** AI Methods I – Day 1: Data cleaning and EDA (IAAC MaAI 2026/27)

## Use case
...

## Dataset
| | |
|---|---|
| **Name** | ... |
| **Source** | [link](...) |
| **Rows × columns (raw)** | ... |
| **Licence** | ... |

## Repository structure
```
├── data/        # downloaded and cleaned data (not committed)
├── figures/     # exported plots
├── notebook/    # data_cleaning_and_EDA.ipynb
...
```

## How to run
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
Open `notebook/data_cleaning_and_EDA.ipynb`, select the `.venv` kernel and run all cells.
The dataset is downloaded automatically by the first cells.

## Data quality summary
| Check | Result | Decision |
|---|---|---|
| Duplicates | ... found / none | ... rows removed / nothing to do |
| Missing values | ... | ... |
| ... | ... | ... |

## Key findings
- ...

![Short description](figures/02_price_distribution.png)

## Limitations
- ...

## License
Code released under the [MIT License](LICENSE). The dataset is subject to its own licence (see above).
````

---

## 5. The other files

### `.gitignore`

Start from GitHub's **Python** template (choose it when creating the repository) and make sure it also contains:

```gitignore
# virtual environments
.venv/
venv/

# Jupyter
.ipynb_checkpoints/

# data: downloaded by the notebook, not committed
data/*
!data/.gitkeep

# OS files
.DS_Store
Thumbs.db

# secrets
.env
kaggle.json
```

⚠️ **Do not ignore `figures/` or `*.png`**: your figures must be committed so the README can show them. To commit a small cleaned dataset (see [section 2](#2-repository-structure)), add an exception such as `!data/my_dataset_clean.csv`.

### `requirements.txt`

Only the packages your notebook imports, **with versions**, for example:

```text
pandas==3.0.6
numpy==2.5.3
matplotlib==3.11.2
seaborn==0.13.2
kagglehub==1.0.2
```

You can generate a full list with `pip freeze > requirements.txt` and then remove what you don't use, or copy the versions from the [course `requirements.txt`](../../requirements.txt). Test it: a fresh environment with `pip install -r requirements.txt` must be able to run your notebook.

### `LICENSE`

A licence tells others what they may do with your **code**. Without one, nobody is legally allowed to reuse it. Use the **MIT License**, like the course repository: on GitHub, choose *Add file → Create new file*, name it `LICENSE`, click *Choose a license template* and pick **MIT License**.

⚠️ Your licence covers **your code only**. The dataset keeps **its own licence**: mention it in the README and respect it (e.g. some datasets do not allow redistribution, which is another reason not to commit them).

---

## 6. Submission checklist

Before you hand in, go through this list:

**Repository**
- [ ] The repository is inside the **iaac-maai** organization, not on a personal profile
- [ ] It has `data/`, `figures/`, `notebook/`, `.gitignore`, `LICENSE`, `README.md` and `requirements.txt`
- [ ] No virtual environment, no large raw data and no credentials are committed (check `git status` and the GitHub page)
- [ ] The final version is pushed to the `main` branch **before 19 October 2026**

**Notebook**
- [ ] It runs from top to bottom without errors after *Restart kernel → Run All*
- [ ] It is committed **with outputs**
- [ ] It loads the data itself and uses **relative paths** only
- [ ] It checks for duplicates, missing values, data types and outliers, and justifies every cleaning decision (or states that none was needed)
- [ ] The EDA includes descriptive statistics, univariate, bivariate, grouped and correlation analysis
- [ ] It answers at least three built-environment questions, each with a plot and a written interpretation
- [ ] It ends with conclusions, limitations and a "ready for modelling" statement
- [ ] It saves the cleaned dataset to `data/` and at least 4 figures to `figures/`

**README**
- [ ] Team, use case, dataset card (with source link and licence), structure, how to run, data quality summary, key findings with at least 2 figures, limitations, licence

**Test it like a stranger:** clone your repository into a new folder, create a fresh environment, install `requirements.txt` and run the notebook. If it fails there, it will fail for us too.

---

## 7. Evaluation

The assignment is graded following the course evaluation criteria (see the syllabus). In particular:

| Criterion | What we look for |
|---|---|
| **Completeness** | All required files, sections and steps are present; the notebook runs |
| **Depth of the EDA** (main weight) | Well-chosen plots and statistics; clear questions answered with evidence; thoughtful interpretations |
| **Technical quality** | Correct use of pandas, matplotlib and seaborn; appropriate data quality checks; no absolute paths |
| **Critical thinking** | Decisions are justified; limitations and biases are discussed; correlation is not confused with causation |
| **Communication** | Clear markdown, readable and labelled figures, a README a stranger can follow |
| **Relevance** | The dataset and the questions are meaningful for architecture and the built environment |
| **Repository conventions** | Clean structure, sensible `.gitignore`, pinned `requirements.txt`, licence |

How dirty your dataset was does **not** affect the grade: a clean dataset with an excellent EDA can get full marks.
