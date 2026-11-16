# HACKATHON SUBMISSIONS

## Assignment: from a trained model to a live app, in five days

Build a **small AI product** for a **built-environment** problem. Start with an **autoencoder** and end with a **live web app** that uses it.

You work in one team repository for the whole week. The repository **grows day by day**. You add one folder per day:

1. **Monday:** train an autoencoder and publish it.
2. **Tuesday:** put the model on a server with an API.
3. **Wednesday:** build an app that calls the model and stores the results.
4. **Thursday:** finish the app and explain how it works.
5. **Friday:** show the app live in the finals.

**Why the repository grows day by day:** the work builds step by step. Each step must work before the next step starts. The Tuesday server needs the Monday model. The Wednesday app needs the Tuesday server. Do **not** let an AI agent generate the whole week at once. A fast start on all days is worth less than a small piece that works today.

This assignment applies what we did in class:
- [`01_autoencoders.ipynb`](../docs/01_autoencoders.ipynb): autoencoders, latent space, anomaly detection, ONNX export.
- [`DATASETS.md`](../docs/DATASETS.md): 24 open datasets for your autoencoder.
- [`model_app/`](../docs/model_app): the class example of a model server (Gradio + ONNX Runtime on Render).
- [`cloudflare_app/`](../docs/cloudflare_app): the class example of an app (Cloudflare Pages + D1 + R2).

| | |
|---|---|
| **Work in** | Teams (as formed in class) |
| **When** | Monday 16 to Friday 20 November 2026, 10:00 to 13:00, room 303 |
| **Instructors** | Salvador Cantuarias Brañes and Serjoscha Duering |
| **Deliverables** | One GitHub repository per team in the [iaac-maai GitHub organization](https://github.com/iaac-maai), one model repository in the [iaac-maai Hugging Face organization](https://huggingface.co/iaac-maai), one model server on Render, one app on Cloudflare |
| **Deadlines** | One per day: **the start of the next class, 10:00**. See the [schedule](#1-schedule-and-deadlines). |
| **How to hand in** | See [section 7 of the course README](../../README.md#7-hand-in-your-submission), and the [mechanics in section 2](#2-how-the-hand-in-works) below |

⚠️ **At each deadline the submission link (submodule) in this repository is updated to the latest commit on your `main` branch.** That commit is your submission for that day. Anything pushed after the deadline is not part of it.

---

## Table of contents

1. [Schedule and deadlines](#1-schedule-and-deadlines)
2. [How the hand-in works](#2-how-the-hand-in-works)
3. [Repository structure](#3-repository-structure)
4. [Monday: the autoencoder](#4-monday-the-autoencoder)
5. [Tuesday: the model server](#5-tuesday-the-model-server)
6. [Wednesday: the app](#6-wednesday-the-app)
7. [Thursday: the final app and the explanation](#7-thursday-the-final-app-and-the-explanation)
8. [Friday: the live demo](#8-friday-the-live-demo)
9. [Impulse talks](#9-impulse-talks)
10. [The other files](#10-the-other-files)
11. [AI usage disclosure](#11-ai-usage-disclosure)
12. [Submission checklists](#12-submission-checklists)
13. [Evaluation](#13-evaluation)

---

## 1. Schedule and deadlines

| Day | Folder | What you deliver | Deadline |
|---|---|---|---|
| **Mon 16 Nov** | `01_autoencoder/` | Dataset, notebook, three setups compared, ONNX model, model on Hugging Face, three slides | Tue 10:00 |
| **Tue 17 Nov** | `02_model_server/` | Model server on Render, README with URL and API example, notebook that calls the API | Wed 10:00 |
| **Wed 18 Nov** | `03_app/` | App on Cloudflare Pages with password, D1 (and R2), sequence diagram | Thu 10:00 |
| **Thu 19 Nov** | `04_final/` | Final app, three diagrams, three statements, finals slides | **Fri 10:00: submission freeze** |
| **Fri 20 Nov** | none | Finals: 5–8 minute presentation with live demo | No new commits count |

⚠️ **Add a folder only after the folder of the day before is submitted.** Do not create `02_model_server/` on Monday.

⚠️ **The trained model must exist by Tuesday 10:00.** The model server needs it. A team without a trained model deploys the **class example model** on Tuesday and swaps in its own model later. Say this in your Tuesday README.

---

## 2. How the hand-in works

You use the same mechanics as Days 1 to 4:

1. Create **one** team repository in the iaac-maai organization on Monday. Use the name pattern:

   ```
   hackathon-team<K>-<topic>
   ```

   For example `hackathon-team3-facade-defects` or `hackathon-team5-energy-anomaly`. Use lowercase letters and hyphens.
2. Push your work to the `main` branch of this repository. You use the **same repository all week**.
3. Send the repository URL to the course maintainers on Monday.
4. The maintainers link it as a **git submodule** in this repository:

   ```
   hackathon/submissions/team_<K>
   ```

5. At each daily deadline, the maintainers **pin** the submodule to the commit that is on `main` at that time.

You do not edit `.gitmodules` or this repository. Section 7 of the course README has the details.

---

## 3. Repository structure

The repository grows. On Thursday it has this structure:

```
hackathon-team<K>-<topic>/
├── 01_autoencoder/
│   ├── README.md                      # Monday: dataset, setup, results, links
│   ├── notebook/
│   │   └── autoencoder.ipynb
│   ├── figures/
│   ├── slides/                        # your three slides (PDF), or a link in the README
│   └── model_app/                     # ONNX model, config, example inputs (as in the class example)
├── 02_model_server/
│   ├── README.md                      # Tuesday: public URL and API example
│   ├── app.py
│   ├── requirements.txt
│   ├── render.yaml
│   └── call_api.ipynb                 # calls the API and prints the result
├── 03_app/
│   ├── README.md                      # Wednesday: public URL, password rule, diagram
│   ├── public/
│   ├── functions/
│   ├── wrangler.toml
│   └── diagrams/
│       └── sequence.png
├── 04_final/
│   ├── README.md                      # Thursday: the final description and the three statements
│   ├── app/                           # the final app (or a link to the final code in 03_app/)
│   └── diagrams/
│       ├── system.png
│       ├── user_flow.png
│       └── sequence.png
├── impulse/
│   └── LINK.md                        # link to your impulse-talk slides (Google Slides)
├── .gitignore
├── LICENSE
└── README.md                          # the front page: team, topic, links, AI usage disclosure
```

| Item | What goes in it |
|---|---|
| `01_autoencoder/` to `04_final/` | One folder per day. Each folder has its own `README.md`. |
| `impulse/` | `LINK.md` with the link to your impulse-talk slides (Google Slides). See [section 9](#9-impulse-talks). |
| `README.md` (root) | The front page. It has the team, the topic, the links to all live parts and the AI usage disclosure. |
| `.gitignore`, `LICENSE` | See [section 10](#10-the-other-files). |

Each folder has its own pinned `requirements.txt` where it has Python code (`01_autoencoder/` and `02_model_server/`).

As before: your GitHub repository holds the **code that produces the results**. Hugging Face holds the **model**. Render and Cloudflare run the **server** and the **app**. Each one links to the others.

⚠️ **Paths:** use relative paths only. In a notebook in `notebook/`, the other folders are one level up.

---

## 4. Monday: the autoencoder

**Folder:** `01_autoencoder/`. **Deadline:** Tuesday 17 November, 10:00 (start of class).

Train an autoencoder on a built-environment dataset and publish it. Do **not** deploy anything today.

### 4.1 What you deliver

1. **A dataset.** Choose one:
   - a dataset from [`DATASETS.md`](../docs/DATASETS.md), or
   - your own data from Days 1 to 4.

   Name the source and the licence in your README.
2. **A notebook** (`notebook/autoencoder.ipynb`). Follow section 10 of notebook 1. The notebook:
   - loads your data and splits it **before** anything else;
   - trains **at least three setups** (for example dense vs convolutional, and two latent sizes) on the **same split and seed**;
   - compares them in **one table**: parameters, training time, validation and test error (and ROC AUC, precision and recall for anomaly detection);
   - says what you conclude, not only the numbers;
   - picks **one of the five uses** and shows **one figure** that proves it works (an anomaly histogram, a similarity row, a denoising row or a VAE blend).
3. **The model, ready for the server.** Export it to **ONNX** as in section 9 of notebook 1. Put the ONNX file, the config (preprocessing and threshold) and a few example inputs in `model_app/`. The ONNX check must give the same scores as PyTorch.
4. **The model on Hugging Face**, with a model card. See [4.2](#42-publish-on-hugging-face).
5. **Three slides** (`slides/` as PDF, or a Google Slides link in the README):
   1. your dataset and the setups you compared (the table);
   2. what you see: one figure and your interpretation;
   3. **one app idea**: who uses it, what they upload or type, what the model answers.
6. **Impulse-talk slides**, only if your topic is on Tuesday. See [section 9](#9-impulse-talks).

### 4.2 Publish on Hugging Face

If you have not set up Hugging Face yet (account, membership of the **iaac-maai** organization, the `hf` command-line tool, an access token and `hf auth login`), follow **[section 5.1 of the Day 2 brief](../../Day_2/submissions/README.md#51-one-time-setup)** first.

**Name** your model repository with this pattern, all lowercase, with hyphens:

```
iaac-maai/hackathon-team<K>-<topic>-ae
```

For example `iaac-maai/hackathon-team3-facade-defects-ae`.

**Create and upload** from the **root of your `01_autoencoder/` folder**, after the notebook has saved the ONNX file and the figures into `models/`:

```bash
hf repos create iaac-maai/hackathon-team3-facade-defects-ae --type model --exist-ok
hf upload iaac-maai/hackathon-team3-facade-defects-ae models . --commit-message "Upload ONNX model and model card"
```

The model card (`models/README.md`) must contain:
- what the model does and for which use case;
- the dataset (source, size, licence);
- the architecture, the latent size and the preprocessing (input size, scaling);
- the training setup and the comparison of setups;
- at least one figure of the training and one figure of the reconstructions;
- the result and its limits: where the model fails;
- a short *How to use* snippet that loads the ONNX file and runs it;
- links to your GitHub repository.

Open the model page and check that the card **renders with all its figures**.

⚠️ **Never put the token in your notebook, your code or your repository.** Do not commit `*.onnx` or `*.pt` files to GitHub. They live on Hugging Face.

### 4.3 The folder README

`01_autoencoder/README.md` must have: the use case, the dataset (name, source, size, licence), how to run, the setups you compared (a table), the result (at least two figures), the link to the Hugging Face model, limitations.

---

## 5. Tuesday: the model server

**Folder:** `02_model_server/`. **Deadline:** Wednesday 18 November, 10:00 (start of class).

Put your model behind an API. Follow the class example in [`hackathon/docs/model_app/`](../docs/model_app). It uses **Gradio** and **ONNX Runtime** and runs on **Render** (free plan).

### 5.1 What you deliver

1. **A model server** that:
   - loads your ONNX model from Hugging Face or from the folder;
   - has a page for people and an API for programs;
   - does the same preprocessing as your Monday notebook.
2. **The server runs on Render** at a public URL.
3. **A README** (`02_model_server/README.md`) with:
   - the **public URL**;
   - what the input is and what the output is;
   - an **API example** that a stranger can copy and run (use `gradio_client`, or plain HTTP as in the class example);
   - a note that the free service sleeps after 15 minutes and needs about one minute to wake up.
4. **A notebook** (`call_api.ipynb`) that calls the API with an example input and **prints the result**.

### 5.2 The model must exist by 10:00

⚠️ **Hard rule: the trained model must exist by Tuesday 10:00.** Tuesday is for the server, not for training.

A team without a trained model deploys the **class example model** from `model_app/` and says so in the README. The team swaps in its own model later. The server must work by Wednesday 10:00.

### 5.3 Keep the model small

Render's free plan has little memory and no GPU. Use small inputs and a small ONNX model. If the build fails, read the build log first. Most errors come from `requirements.txt` or from a wrong file path.

---

## 6. Wednesday: the app

**Folder:** `03_app/`. **Deadline:** Thursday 19 November, 10:00 (start of class).

Build an app that calls your model server and stores the results. Follow the class example in [`hackathon/docs/cloudflare_app/`](../docs/cloudflare_app). It uses **Cloudflare Pages**, **Pages Functions**, **D1** and **R2**.

### 6.1 What you deliver

1. **An app on Cloudflare Pages** at a public URL that:
   - is **protected by a password**;
   - lets a user send an input to your model server (through a Pages Function);
   - shows the result;
   - **stores each result in D1**;
   - stores uploaded files in **R2**, if your app uses files.
2. **A README** (`03_app/README.md`) with:
   - the **public URL**;
   - how a user logs in (tell us the password in the README of the *course hand-in*, not in the repository: see the warning below);
   - what the app does, in a few sentences;
   - which Cloudflare services the app uses and why.
3. **A sequence diagram** of the app (`diagrams/sequence.png`). It shows who talks to whom, in order: user, page, function, model server, D1, R2.

⚠️ **Never commit secrets.** The password is a Cloudflare *secret*. Locally it lives in `.dev.vars`. `.dev.vars` is in `.gitignore` and is **never** committed. Do not put tokens or passwords in `wrangler.toml`.

⚠️ **Free plans have limits.** Check the model server URL in `wrangler.toml` (`MODEL_URL`, without a slash at the end). Tell us in the README if the first call is slow because the Render server is asleep.

---

## 7. Thursday: the final app and the explanation

**Folder:** `04_final/`. **Deadline:** Friday 20 November, 10:00 (start of class). **This is the submission freeze.**

Finish the app and explain it. Today you do **not** start new features that you cannot finish. A smaller app that works is better than a large app that does not.

### 7.1 What you deliver

1. **The final app**, live at its public URL. It uses your own model, your model server and D1 (and R2 if used).
2. **Three diagrams** in `04_final/diagrams/`:
   - a **system diagram**: the parts of the system and how they connect (Hugging Face, Render, Cloudflare Pages, D1, R2);
   - a **user-flow diagram**: what a user does, step by step, and what the user sees;
   - a **sequence diagram**: who talks to whom, in order, for one request.
3. **Three statements** in `04_final/README.md`, each a short paragraph:
   - **What is clear:** what works and what you understand well.
   - **What is still unclear:** what you do not understand, or what you did not test.
   - **What you would do next:** the next three steps if the project continued.
4. **The final README** with: the use case, the live URLs (app, model server, model), the diagrams, the three statements, limitations.

⚠️ **Write the three statements yourselves.** We ask about them in the finals. Be honest: a clear "we do not know yet" is worth more than a vague claim.

---

## 8. Friday: the live demo

On Friday every team presents for **5 to 8 minutes**, with a **live demo** of its app. We send a slide template before Friday.

Show:
- the idea and the model: what your autoencoder does and what you found;
- your diagrams: system, user flow and sequence;
- the services you use and how they connect;
- where your data is stored, and what is persistent;
- the live app.

- Use the **public URLs**, not a local copy.
- Wake up your Render server before you start.
- **No new commits count.** The pinned commit of Friday 10:00 is your submission.

---

## 9. Impulse talks

Each team gives **one impulse talk** of **3 to 5 minutes**, at the start of a class. Each topic is a curious question: **how does that work?** Show a real use case, then explain how it works inside with a **system diagram** or a **sequence diagram**.

| Day | Topics (one per team) |
|---|---|
| **Tuesday** | Latent Space, Generative AI, Anomaly Detection |
| **Wednesday** | App Anatomy, Cloud Services, Data Storage |
| **Thursday** | Agent Skills, Sign in with Google, Passwords, 1,000 users at once |

- Your team chooses a topic on Monday.
- Make the slides in **Google Slides** and share them as "anyone with the link can view".
- Put the link in `impulse/LINK.md`. It is due at the start of the class in which you present.
- The structure, the questions and the diagram types are in [`IMPULSE_TALKS.md`](../docs/IMPULSE_TALKS.md).
---

## 10. The other files

### `.gitignore`

```gitignore
# virtual environments
.venv/
venv/

# Jupyter
.ipynb_checkpoints/

# data: downloaded by the notebook, not committed
data/*
!data/.gitkeep

# model files: on Hugging Face, not on GitHub
*.onnx
*.pt
*.joblib

# Cloudflare and Node
node_modules/
.wrangler/

# OS files
.DS_Store
Thumbs.db

# secrets: NEVER commit these
.env
.dev.vars
kaggle.json
```

⚠️ Do **not** ignore `figures/`, `*.png` or the model card `models/README.md`. They must be on GitHub.

### `requirements.txt`

Only the packages your code imports, **with versions**. Use one file per folder with Python code. For example:

```text
onnxruntime==<version>
gradio==<version>
numpy==<version>
```

Use the versions from the class example as a start: `gradio==6.32.0`, `onnxruntime==1.31.1` (see `hackathon/docs/model_app/requirements.txt`).

### `LICENSE`

**MIT License** for your code. The data keeps **its own licence**: say it in the README and in the model card.

### Big files

Models and large data go to **Hugging Face**, not to GitHub. Do not commit files bigger than a few MB (GitHub warns at 50 MB and blocks files over 100 MB).

### Secrets

Never commit passwords, tokens or API keys. This includes `.dev.vars`, `.env` and your Hugging Face token. If you commit a secret by mistake, **change the secret at once**. Deleting the file is not enough: git keeps the history.

---

## 11. AI usage disclosure

The AI usage disclosure is **required** and **graded**. Put it in the **root `README.md`** of your repository. It covers the whole week.

Answer these questions honestly:

- **Tools:** which AI tools did you use (for example Claude Code, ChatGPT, Copilot)? Write "none" if you used none.
- **Why:** why did you use them?
- **Where:** for which parts (data, notebook, server, app, diagrams, text)?
- **How:** how did you use them? What did you ask? What did you accept, change or reject?
- **Lessons learned:** what worked, what did not, what would you do differently?

If you did not use AI tools, write one sentence saying so.

⚠️ You are responsible for everything in your submission, including what an AI tool produced. **You must be able to explain every part in the finals.** The three statements on Thursday must be your own.

A starting template:

````markdown
## AI usage disclosure
| | |
|---|---|
| **Tools** | e.g. Claude Code (agent), ChatGPT (chat) / none |
| **Why** | ... |
| **Where** | Monday: ... Tuesday: ... Wednesday: ... Thursday: ... |
| **How** | ... |

**Lessons learned:** ...
````

---

## 12. Submission checklists

### Monday: `01_autoencoder/`

**GitHub**
- [ ] The repository is inside the **iaac-maai** organization and is named `hackathon-team<K>-<topic>`
- [ ] It has `01_autoencoder/` (with `README.md`, `notebook/`, `figures/`, `slides/`, `model_app/`), `.gitignore`, `LICENSE` and a root `README.md`
- [ ] It has **no** other day folder yet
- [ ] No data, model files, virtual environment or credentials are committed
- [ ] The final version is pushed to `main` before the next class starts (10:00)

**Notebook**
- [ ] It runs from top to bottom and is committed **with outputs**
- [ ] It splits the data first and trains on **your** data; the validation loss goes down too
- [ ] At least three setups in one table, on the same split and seed, with a written conclusion
- [ ] One of the five uses, with one figure that proves it works
- [ ] ONNX model, config and example inputs are in `model_app/`, and the ONNX check matches PyTorch
- [ ] You can say in one sentence what the model's output means for a user

**Hugging Face**
- [ ] The model is `iaac-maai/hackathon-team<K>-<topic>-ae` with the ONNX file and a model card
- [ ] The card renders with its figures, and has a *How to use* snippet
- [ ] The card links to GitHub, and the GitHub README links to the model

**Talk**
- [ ] Three slides: dataset and table, figure and interpretation, one app idea
- [ ] Tuesday topic only: the link to your impulse-talk slides is in `impulse/LINK.md`

### Tuesday: `02_model_server/`

- [ ] The model server runs on **Render** at a public URL
- [ ] It uses your own model, or the class example model (and the README says so)
- [ ] The README has the public URL, the input and output, and an API example
- [ ] The README says that the free service sleeps
- [ ] `call_api.ipynb` calls the API and prints the result, with outputs
- [ ] `requirements.txt` has pinned versions
- [ ] No secrets are committed
- [ ] The final version is pushed to `main` before the next class starts (10:00)

### Wednesday: `03_app/`

- [ ] The app runs on **Cloudflare Pages** at a public URL
- [ ] The app is **protected by a password**, and the password is a secret
- [ ] The app calls your model server and shows the result
- [ ] Each result is stored in **D1** (files in **R2**, if used)
- [ ] The README has the public URL and says what the app does
- [ ] `diagrams/sequence.png` shows the app
- [ ] `.dev.vars` is **not** committed
- [ ] The final version is pushed to `main` before the next class starts (10:00)

### Thursday: `04_final/`

- [ ] The final app is live and works end to end
- [ ] `diagrams/` has a **system diagram**, a **user-flow diagram** and a **sequence diagram**
- [ ] `04_final/README.md` has the three statements: **what is clear**, **what is still unclear**, **what you would do next**
- [ ] The root `README.md` links to the app, the server and the model
- [ ] The **AI usage disclosure** is complete and honest
- [ ] No secrets, data or model files are committed
- [ ] The final version is pushed to `main` before **Friday 10:00** (submission freeze)

### Friday

- [ ] You tested the live URLs on the day of the finals
- [ ] The Render server is awake before the demo

---

## 13. Evaluation

The assignment is graded following the course evaluation criteria (see the syllabus). In particular:

| Criterion | What we look for |
|---|---|
| **Working pieces each day** | Each daily milestone is met: the model exists on Monday, the server runs on Tuesday, the app runs on Wednesday. Each step works before the next step starts. |
| **The final app is live** | The final app runs at its public URL, uses your own model through your own server, stores results, and is protected by a password. It works in the live demo. |
| **Diagrams and three statements** | The system, user-flow and sequence diagrams are correct and match the real app. The three statements (what is clear, what is still unclear, what you would do next) are honest and specific. |
| **Understanding** | You can explain what the autoencoder learned, how the parts connect, and where and why the system fails. You can explain every part that an AI tool wrote. |
| **AI usage disclosure** | The disclosure is complete, honest and useful. It names what the AI did and what the team checked and changed. |
| **Repository hygiene** | One team repository with the right structure, one folder per day, README per folder, pinned `requirements.txt`, `.gitignore`, licence, correct Hugging Face naming. No secrets, no big files on GitHub. |

A small system that works from end to end and that you understand is worth more than a large system that nobody can run or explain.
