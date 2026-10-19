# DAY_2 SUBMISSIONS

## Assignment: a supervised learning pipeline, published as a model

Train, evaluate and **publish** a supervised machine learning model for a **built-environment** problem.

You deliver two things:

1. A **GitHub repository** with a documented notebook that goes from data to an honestly evaluated model, structured like the Day 1 repository.
2. The **trained model on Hugging Face**, inside the course organization, with a **model card**: a README that tells anyone what the model does, what it was trained on, how well it works and where it fails.

This assignment applies what we did in class ([`Day_2/docs/application_of_supervised_learning.ipynb`](../docs/application_of_supervised_learning.ipynb)) to your own data. The natural starting point is the **cleaned dataset from your Day 1 assignment**.

| | |
|---|---|
| **Work in** | Teams (as formed in class) |
| **Deliverables** | One GitHub repository in the [iaac-maai GitHub organization](https://github.com/iaac-maai) **and** one model repository in the [iaac-maai Hugging Face organization](https://huggingface.co/iaac-maai) |
| **Deadline** | **Monday, 26 October 2026, 23:59 CET** |
| **How to hand in** | See [section 7 of the course README](../../README.md#7-hand-in-your-submission), and put the Hugging Face link in your GitHub README |

⚠️ **On 26 October 2026 the submission links (submodules) in this repository are updated to the latest commit on your `main` branch.** That commit, and the version of your Hugging Face model at that time, is your submission. Anything pushed afterwards is not part of it.

---

## Table of contents

1. [Choose a problem](#1-choose-a-problem)
2. [Repository structure](#2-repository-structure)
3. [The notebook](#3-the-notebook)
4. [The GitHub README](#4-the-github-readme)
5. [Publish your model on Hugging Face](#5-publish-your-model-on-hugging-face)
6. [The model card](#6-the-model-card)
7. [The other files](#7-the-other-files)
8. [Submission checklist](#8-submission-checklist)
9. [Evaluation](#9-evaluation)

---

## 1. Choose a problem

Use the **cleaned dataset from your Day 1 assignment** and pick a **target variable** to predict:

| Problem type | Target | Examples |
|---|---|---|
| **Regression** | a number | price, energy demand (kWh/m²), construction cost, CO₂ emissions, occupancy |
| **Classification** | a category | energy rating, building type, "permit approved: yes / no", element type in a BIM schedule |

If your Day 1 dataset has no meaningful target, you may switch to another tabular built-environment dataset (see the [Day 1 brief](../../Day_1/submissions/README.md#1-choose-a-dataset) for where to look). Explain why in your README.

Before you start, answer three questions. They shape every later decision and they go in your README and model card:

1. **Who would use this model, and for what?** For example: "an architect testing massing options in a feasibility study".
2. **Which mistake is more expensive?** Over- or under-estimating? Missing a positive or raising a false alarm? This choice determines your **main metric**.
3. **What would the user know at prediction time?** Only those columns may be inputs. Everything else is **leakage**.

⚠️ **Do not use the two datasets from class** (house prices and energy efficiency). If you are unsure whether your problem is suitable, ask the course maintainers before you start.

---

## 2. Repository structure

Your repository must contain exactly this structure:

```
<repo-name>/
├── data/
│   └── .gitkeep                     # keeps the folder in git
├── figures/
│   ├── 01_model_comparison.png      # your exported plots
│   └── ...
├── models/
│   ├── README.md                    # the Hugging Face model card (committed)
│   └── model.joblib                 # the trained pipeline (NOT committed: uploaded to Hugging Face)
├── notebook/
│   └── supervised_learning.ipynb
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

| Item | What goes in it |
|---|---|
| `data/` | Same rules as Day 1: raw data is downloaded by the notebook, not committed. A small cleaned dataset (a few MB) may be committed. |
| `figures/` | The key plots, exported by the notebook as `.png` files, **committed** and shown in your README. |
| `models/` | Exactly what you upload to Hugging Face: the model card (`README.md`, committed to GitHub) and the trained pipeline (`model.joblib`, **only** on Hugging Face). |
| `notebook/` | Your notebook (one main notebook; extra notebooks are allowed if you explain them in the README). |
| `.gitignore`, `LICENSE`, `requirements.txt` | See [section 7](#7-the-other-files). |
| `README.md` | The front page of your project. See [section 4](#4-the-github-readme). |

Why is the model file not on GitHub? Model files can be large, change every time you retrain, and are not readable code. Hugging Face is built to host and version them, and makes them easy to download. Your GitHub repository holds the **code that produces the model**; Hugging Face holds the **model itself**. Each one links to the other.

⚠️ **Paths:** the notebook lives in `notebook/`, so the other folders are **one level up**. Use relative paths only:

```python
from pathlib import Path

DATA_DIR = Path("..") / "data"
FIG_DIR = Path("..") / "figures"
MODEL_DIR = Path("..") / "models"
```

---

## 3. The notebook

The notebook must **run from top to bottom without errors** (*Restart kernel → Run All*) and must be committed **with its outputs**. As on Day 1, every step needs a **markdown cell that explains what you do and why**. Here, the decisions that matter most are **how you evaluate**: which metric, which baseline, how you avoided leakage, and why you chose the final model.

### 3.1 Required sections

**0. Introduction and problem definition**
- Title, team members, and the **use case**: who uses the model, for what, and which mistake is costly.
- Load the data (the cleaned Day 1 dataset, or download it).
- State the **problem type** (regression or classification), the **target**, and your **main metric**, with a short justification.

**1. Features, leakage and split**
- Define `X` and `y`. List the columns you **removed** as inputs and **why** (identifiers, leakage, columns not known at prediction time).
- For a skewed regression target (prices, areas…), consider predicting its **log**, and explain your choice.
- **Train/test split** with a fixed `random_state`; **stratified** (`stratify=y`) for classification. The test set stays locked until section 5.

**2. Preprocessing pipeline**
- Build the preprocessing with a scikit-learn **`Pipeline`** and **`ColumnTransformer`**: imputation, scaling where needed, and **encoding** of categorical columns.
- **Justify each encoder**: one-hot for nominal categories, ordinal encoding *with an explicit order* for ordinal ones. Watch out for categories stored as numbers.

**3. Model comparison**
- A **baseline** (`DummyRegressor` / `DummyClassifier`).
- At least **three model families**, including **one linear model** and **one tree ensemble** (random forest or gradient boosting).
- **5-fold cross-validation** on the training set only, with the **same folds and metrics** for every model.
- A results **table** (mean ± standard deviation) and a **chart**. Comment on the ranking: which differences are real, and which are within noise?

**4. Diagnosis and tuning**
- At least **one** of these: a **validation curve** of a complexity hyperparameter (e.g. `max_depth`), or a **`GridSearchCV`** on your best model.
- Say whether the model under- or overfits, and whether the gain from tuning is worth it.

**5. Final evaluation on the test set (once!)**
- Retrain the chosen pipeline on the full training set and evaluate it **once** on the test set. Compare with the cross-validation score.
- **Regression:** MAE, RMSE and R² (MAPE if useful), a **predicted vs actual** plot, and a look at the **errors** (distribution, or per group).
- **Classification:** the **confusion matrix**, a `classification_report` (precision, recall, F1 per class), and accuracy **only together with** these. Discuss which classes get confused.

**6. Interpretation**
- Which features matter? Use **permutation importance** (any model) and/or **coefficients** (linear models).
- Read the results as a designer or engineer: what does the model rely on, and does it make sense? Where does it fail?

**7. Save and reuse the model**
- Save the **whole fitted pipeline** (preprocessing + model) to `models/model.joblib` with `joblib.dump`.
- **Reload** it and predict for **one hand-made example** (e.g. a new design option), as in class.

**8. Conclusions**
- The **key findings** (3–6 bullet points, with numbers).
- The **final model** you chose and **why** (accuracy, speed, interpretability).
- **Limitations and biases**: data, evaluation, and where the model should **not** be used.

### 3.2 Technical requirements

- Libraries: **pandas**, **numpy**, **matplotlib**/**seaborn** and **scikit-learn**.
- All preprocessing inside a **`Pipeline`**, so nothing leaks during cross-validation.
- The **test set is used once**, after all choices are made.
- A fixed `random_state` everywhere randomness is involved.
- At least **4 key figures** exported to `figures/` (e.g. model comparison, validation curve, predicted vs actual or confusion matrix, feature importance).
- **No credentials** in the notebook: no Kaggle keys, **no Hugging Face tokens**.

---

## 4. The GitHub README

Someone who has never seen your project should understand it and be able to run it **in under five minutes**. It must contain:

1. **Title** and **team members**.
2. **Use case**: who uses the model, for what, and which mistake is costly (2–4 sentences).
3. **Model on Hugging Face**: the link to your model repository, at the top, impossible to miss.
4. **Dataset**: name, link, short description, licence.
5. **Repository structure**: the folder tree with one line per item.
6. **How to run**: environment, `requirements.txt`, data, notebook.
7. **Approach**: target, features, preprocessing and encoders, models compared (a short paragraph or table).
8. **Results**: the model comparison table (cross-validation, mean ± std, including the baseline) and the **final test scores**, with **at least 2 figures** embedded from `figures/`.
9. **Limitations**.
10. **AI usage disclosure**: same rules as Day 1, see [below](#ai-usage-disclosure).
11. **License**.

### AI usage disclosure

Using AI tools is allowed, but it must be disclosed. As on Day 1, this section is also a way to **share with the class how to use agentic tools effectively**, so be concrete:

- **Which tools** did you use (e.g. ChatGPT, Claude, Claude Code, GitHub Copilot, Cursor)? Did **agents** edit files or run code by themselves?
- **Why**: what did you expect the tool to help with?
- **Where**: which parts (pipeline code, model comparison, plots, model card, README)?
- **How**: your workflow. What did you ask for (an example prompt helps), how did you check the output, and what did you correct or reject?
- **Lessons learned**: what worked, what did not, what would you do differently?

If you did not use AI tools, write one sentence saying so.

⚠️ You are responsible for everything in your submission, including what an AI tool produced. Every modelling decision and every metric must be one your team understands and can defend.

A starting template:

````markdown
# <Project title>

**Team:** Name Surname, Name Surname, Name Surname
**Course:** AI Methods I – Day 2: Supervised learning (IAAC MaAI 2026/27)
**Model on Hugging Face:** https://huggingface.co/iaac-maai/<model-repo>

## Use case
...

## Dataset
| | |
|---|---|
| **Name** | ... |
| **Source** | [link](...) |
| **Rows used** | ... |
| **Licence** | ... |

## Repository structure
```
├── data/        # downloaded and cleaned data (not committed)
├── figures/     # exported plots
├── models/      # model card (README.md); model.joblib lives on Hugging Face
├── notebook/    # supervised_learning.ipynb
...
```

## How to run
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
Open `notebook/supervised_learning.ipynb`, select the `.venv` kernel and run all cells.

## Approach
- **Target:** ... (regression / classification), main metric: ... because ...
- **Features:** ... (removed: ... because ...)
- **Preprocessing:** ...
- **Models compared:** baseline, ..., ..., ...

## Results
| Model | CV score (mean ± std) | ... |
|---|---|---|
| Baseline | ... | |
| ... | ... | |

**Final model:** ... **Test set:** ...

![Model comparison](figures/01_model_comparison.png)

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

## 5. Publish your model on Hugging Face

[Hugging Face](https://huggingface.co) is the main platform for sharing machine learning models: like GitHub, but for models and datasets. Every model lives in a **model repository** with its files and a **model card** (`README.md`). You upload with the **`hf` command-line tool (CLI)**.

### 5.1 One-time setup

**1. Account and organization.** Create a Hugging Face account (if you do not have one) and **join the [iaac-maai organization](https://huggingface.co/iaac-maai)**: accept the invitation from the course maintainers, or click *Request to join* on the organization page and tell us your username. You can only upload to the organization once you are a member with write access.

**2. Install the `hf` CLI.** Use **one** of these:

```bash
# Option A: inside your project's virtual environment (recommended)
pip install -U huggingface_hub
```

```bash
# Option B: standalone installer (macOS / Linux)
curl -LsSf https://hf.co/cli/install.sh | bash -s
```

Check it works:

```bash
hf version
```

> The old command `huggingface-cli` is deprecated. Use `hf`.

**3. Create an access token.** Go to [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) → *Create new token*. Choose a **fine-grained** token, and under *Organizations* give it **write** access to repositories in **iaac-maai** (a classic *Write* token also works). Copy the token: it is shown only once.

**4. Log in.**

```bash
hf auth login
```

Paste the token when asked. Then check who you are and that **iaac-maai** appears in your organizations:

```bash
hf auth whoami
```

⚠️ **Never put the token in your notebook, your code or your repository.** `hf auth login` stores it safely on your computer, outside the project. If a token is ever pushed to GitHub, delete it immediately on the tokens page and create a new one.

### 5.2 Name your model repository

Use this pattern, all lowercase, with hyphens:

```
iaac-maai/day2-<team>-<what-it-predicts>
```

For example `iaac-maai/day2-team3-energy-class` or `iaac-maai/day2-team5-construction-cost`.

### 5.3 Create the repository and upload

Run these commands from the **root of your GitHub repository**, after the notebook has saved `models/model.joblib` and you have written `models/README.md` (the model card, [section 6](#6-the-model-card)):

```bash
# 1. Create the model repository inside the organization
hf repos create iaac-maai/day2-team3-energy-class --type model --exist-ok

# 2. Upload the contents of models/ (model.joblib + README.md) to the root of the repository
hf upload iaac-maai/day2-team3-energy-class models . --commit-message "Upload trained pipeline and model card"
```

`hf upload <repo-id> <local-folder> <path-in-repo>` uploads a whole folder; `.` means "at the root of the repository". Each upload creates a new **commit** on Hugging Face, so the history of your model is kept, just like on GitHub.

Open `https://huggingface.co/iaac-maai/<your-repo>` to check that the files are there and that the model card renders correctly.

**Updating the model** (after retraining, or after improving the model card): run the same `hf upload` command again with a new commit message. To update only the model card:

```bash
hf upload iaac-maai/day2-team3-energy-class models/README.md README.md --commit-message "Improve model card"
```

**Visibility:** model repositories are **public** by default, which is what we want: a published model is part of your portfolio. If your training data has a licence that forbids sharing derived models, tell the course maintainers before uploading.

### 5.4 Test it like a stranger

Download your model into a **fresh folder** and load it, to prove that anyone can use it:

```bash
hf download iaac-maai/day2-team3-energy-class --local-dir test_download
```

```python
import joblib
import pandas as pd

model = joblib.load("test_download/model.joblib")
example = pd.DataFrame([{ ... }])   # one row with the same columns as your training data
print(model.predict(example))
```

If this fails (missing package, wrong scikit-learn version, missing column), fix it before the deadline. Delete `test_download/` afterwards; do not commit it.

---

## 6. The model card

The model card is the `README.md` of your Hugging Face repository (written in `models/README.md`, committed to GitHub, uploaded to Hugging Face). It is the **user manual of your model**: someone should be able to decide whether to use it, and how, **without opening your notebook**. A model without a card is a black box nobody should trust.

It has two parts: a **metadata header** (YAML between `---` lines, read by Hugging Face to make your model searchable) and the **text**.

### 6.1 What it must contain

| Section | What to write |
|---|---|
| **Metadata header** | Licence, library (`sklearn`), task (`tabular-regression` / `tabular-classification`), tags, metrics |
| **Model summary** | One paragraph: what it predicts, from what, for whom, with which algorithm |
| **Intended use** | Who should use it and for what. **Out-of-scope uses**: where it must *not* be used (other countries, building types, value ranges…) |
| **Model details** | Problem type, algorithm and key **hyperparameters**, preprocessing steps, file format, scikit-learn version, authors |
| **Input features** | A table: every input column, its type, unit, description, and how it is preprocessed. The **exact column names** the model expects |
| **Output** | What the model returns: unit (and log transform if any) or the list of classes |
| **Training data** | Dataset name, **source link**, licence, size, what one row represents, period / location covered, a summary of the cleaning (with a link to your Day 1 repository if relevant) |
| **Training procedure** | Train/test split (sizes, stratification, seed), cross-validation, **models compared** (table with the baseline), tuning, why this model was chosen |
| **Evaluation** | Test-set metrics, cross-validation mean ± std, baseline for comparison; for classification the confusion matrix or per-class scores. Where does it perform worst? |
| **Limitations and biases** | What the data does not represent, known failure cases, leakage risks, assumptions. Be honest: this is the most important section for a user |
| **How to use** | A short, working Python snippet: download, load, predict on one example |
| **Reproducibility** | Link to the **GitHub repository** with the full pipeline, the notebook, and the `requirements.txt` / library versions |
| **Security note** | `.joblib` files are pickles: they can run code when loaded. Users should only load models from sources they trust |
| **Authors, course and licence** | Team members, "AI Methods I, IAAC MaAI 2026/27", licence of the model and of the data |

### 6.2 Template

Copy this into `models/README.md` and fill in every `...`. Delete the comments (`<!-- -->`) when you are done.

````markdown
---
license: mit
library_name: sklearn
pipeline_tag: tabular-regression        # or: tabular-classification
tags:
  - sklearn
  - tabular-regression                  # or: tabular-classification
  - built-environment
  - iaac-maai
  - ai-methods-1
metrics:
  - r2                                  # classification: accuracy, f1 ...
  - mae
---

# <Model name>: predicting <target> from <inputs>

<!-- One paragraph: what it predicts, from what, for whom, with which algorithm. -->
This model predicts ... from ... It is a scikit-learn pipeline (preprocessing + <algorithm>) trained on ... It was built as a course assignment for AI Methods I (IAAC MaAI 2026/27).

- **GitHub repository (full pipeline):** https://github.com/iaac-maai/<repo-name>
- **Team:** Name Surname, Name Surname, Name Surname

## Intended use

- **Intended users and use:** ...
- **Out-of-scope uses:** ... <!-- e.g. other countries, building types or value ranges not in the training data -->

## Model details

| | |
|---|---|
| **Problem type** | Regression / classification |
| **Algorithm** | e.g. HistGradientBoostingRegressor |
| **Key hyperparameters** | e.g. learning_rate=0.1, max_leaf_nodes=63 |
| **Preprocessing** | e.g. median imputation + scaling (numeric), one-hot encoding (nominal), ordinal encoding (ordinal) |
| **File** | `model.joblib`: the whole fitted scikit-learn `Pipeline` |
| **Library versions** | scikit-learn ..., pandas ..., numpy ..., Python ... |

## Input features

The model expects a `pandas.DataFrame` with exactly these columns:

| Column | Type | Unit | Description | Preprocessing |
|---|---|---|---|---|
| ... | numeric | m² | ... | median imputation, scaling |
| ... | category | – | ... | one-hot encoding |

## Output

<!-- Regression: unit, and whether the model predicts a log value that must be converted back.
     Classification: the list of classes and what they mean. -->
...

## Training data

| | |
|---|---|
| **Dataset** | ... |
| **Source** | [link](...) |
| **Licence** | ... |
| **One row is** | ... |
| **Size used** | ... rows (train ...; test ...) |
| **Coverage** | location, period, building types... |

**Cleaning:** ... <!-- short summary, link to your Day 1 repository if relevant -->

## Training procedure

- **Split:** 80 / 20 train/test, `random_state=42` (stratified for classification)
- **Validation:** 5-fold cross-validation on the training set
- **Models compared:**

| Model | CV <metric> (mean ± std) |
|---|---|
| Baseline | ... |
| ... | ... |

- **Tuning:** ...
- **Why this model:** ...

## Evaluation

Results on the held-out **test set** (used once):

| Metric | Test | Cross-validation |
|---|---|---|
| ... | ... | ... ± ... |

**Where it performs worst:** ...

## Limitations and biases

- ...

## How to use

```python
import joblib
import pandas as pd
from huggingface_hub import hf_hub_download

path = hf_hub_download(repo_id="iaac-maai/<model-repo>", filename="model.joblib")
model = joblib.load(path)

example = pd.DataFrame([{
    "column_1": ...,
    "column_2": ...,
}])
print(model.predict(example))
```

⚠️ `.joblib` files are Python pickles and can execute code when loaded: only load models from sources you trust. Use the library versions listed above; the model may not load with other versions.

## Reproducibility

The full pipeline (data loading, cleaning, training, evaluation) is in the GitHub repository: https://github.com/iaac-maai/<repo-name>. Run `notebook/supervised_learning.ipynb` with the packages in `requirements.txt`.

## Authors and licence

Name Surname, Name Surname, Name Surname. Course assignment for **AI Methods I: Machine Learning for the Built Environment**, IAAC MaAI 2026/27.

Model released under the MIT License. The training data is subject to its own licence: ...
````

💡 **Tip:** to show a figure in the model card, copy it into `models/` (e.g. `models/confusion_matrix.png`), embed it with `![Confusion matrix](confusion_matrix.png)` and upload again.

---

## 7. The other files

### `.gitignore`

Same as Day 1 (see the [Day 1 brief](../../Day_1/submissions/README.md#5-the-other-files)), plus the model file and any test downloads:

```gitignore
# virtual environments
.venv/
venv/

# Jupyter
.ipynb_checkpoints/

# data: downloaded by the notebook, not committed
data/*
!data/.gitkeep

# trained model: published on Hugging Face, not on GitHub
models/*.joblib
test_download/

# OS files
.DS_Store
Thumbs.db

# secrets
.env
kaggle.json
```

⚠️ Do **not** ignore `figures/`, `*.png` or `models/README.md`: they must be on GitHub.

### `requirements.txt`

Only the packages your notebook imports, **with versions**. This assignment adds scikit-learn, joblib and the Hugging Face library, for example:

```text
pandas==3.0.6
numpy==2.5.3
matplotlib==3.11.2
seaborn==0.13.2
scikit-learn==1.9.1
joblib==1.6.0
huggingface_hub
kagglehub==1.0.2
```

Pin **scikit-learn** exactly: a saved model may not load with another version, and your model card refers to these versions.

### `LICENSE`

**MIT License**, as on Day 1. Use the same licence in the model card's metadata (`license: mit`). The training data keeps **its own licence**: mention it in both READMEs.

---

## 8. Submission checklist

**GitHub repository**
- [ ] The repository is inside the **iaac-maai** GitHub organization
- [ ] It has `data/`, `figures/`, `models/` (with the model card), `notebook/`, `.gitignore`, `LICENSE`, `README.md` and `requirements.txt`
- [ ] No virtual environment, raw data, `.joblib` model file or credentials are committed
- [ ] The final version is pushed to `main` **before 26 October 2026**

**Notebook**
- [ ] It runs from top to bottom without errors and is committed **with outputs**
- [ ] It states the use case, target, problem type and main metric, with a justification
- [ ] It lists the removed input columns and why (leakage)
- [ ] All preprocessing is in a `Pipeline` / `ColumnTransformer`, with justified encoders
- [ ] It compares a **baseline** and at least **three model families** (one linear, one tree ensemble) with **5-fold cross-validation**, reported as mean ± std
- [ ] It includes a validation curve or a `GridSearchCV`
- [ ] The **test set is used once**, with the right metrics and plots for the problem type
- [ ] It interprets the model (feature importance) and discusses where it fails
- [ ] It saves the pipeline to `models/model.joblib`, reloads it and predicts on a hand-made example
- [ ] It exports at least 4 figures to `figures/`

**Hugging Face model**
- [ ] The model repository is inside the **iaac-maai** Hugging Face organization and named `day2-<team>-<what-it-predicts>`
- [ ] It contains `model.joblib` and a complete **model card** (every section of [6.1](#61-what-it-must-contain))
- [ ] The model card links to the GitHub repository, and the GitHub README links to the model
- [ ] It loads and predicts in a fresh folder ([section 5.4](#54-test-it-like-a-stranger))

**README**
- [ ] Team, use case, Hugging Face link, dataset, structure, how to run, approach, results with at least 2 figures, limitations, AI usage disclosure, licence

---

## 9. Evaluation

The assignment is graded following the course evaluation criteria (see the syllabus). In particular:

| Criterion | What we look for |
|---|---|
| **Completeness** | All required files, sections and steps are present; the notebook runs; the model is published and loads |
| **Evaluation methodology** (main weight) | Honest evaluation: baseline, cross-validation, pipelines, no leakage, test set used once, metrics that match the use case |
| **Model choice** | Several model families compared fairly; the final choice is justified (accuracy, speed, interpretability) |
| **Critical thinking** | Interpretation of the features; where the model fails; limitations and biases; correlation is not confused with causation |
| **Communication** | Clear notebook, README and **model card** that a stranger can follow; readable, labelled figures; an honest and useful AI usage disclosure |
| **Relevance** | The problem and the model are meaningful for architecture and the built environment |
| **Repository and publication conventions** | Clean structure, sensible `.gitignore`, pinned `requirements.txt`, licence, correct Hugging Face naming and metadata |

A model with modest scores and an honest, well-argued evaluation is worth more than an impressive score that cannot be trusted.
