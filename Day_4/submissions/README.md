# DAY_4 SUBMISSIONS

## Assignment: a trained neural network, published as a model

Train a **neural network** for a **built-environment** problem, check what it learned, and **publish** it so that others can see what it does and use it.

You choose one of two tracks:

- **Track A: train your own network.** An MLP on your tabular data (**A1**), or an image classifier fine-tuned from a pretrained CNN (**A2**).
- **Track B: fine-tune a detection or segmentation pipeline.** A YOLO detector (or segmenter) fine-tuned on built-environment images, optionally combined with SAM.

You deliver:

1. A **GitHub repository** with a documented notebook that goes from data to an honestly evaluated network, structured like the Day 2 and Day 3 repositories.
2. The **trained model on Hugging Face**, with a **model card** that **shows**, with figures, how the model was trained, how well it works and what it does.
3. **Track B with images you annotated yourselves only:** the **annotated dataset on Hugging Face**, with a dataset card.

**The principle of this assignment: show the model before anyone uses it.** Nobody should have to download your model to find out what it does or how good it is. Your **README** and your **model card** must each contain **at least one figure, and preferably several**, that show the **training process**, the **performance** and **what the model actually does** (see [section 4](#4-show-the-model-figures)).

This assignment applies what we did in class to your own problem:
- [`01_nn_fundamentals_and_tensors.ipynb`](../docs/01_nn_fundamentals_and_tensors.ipynb): tensors, MLPs, the training loop, looking inside a network, regularisation (Track A).
- [`02_computer_vision_opencv.ipynb`](../docs/02_computer_vision_opencv.ipynb): images as arrays, rule-based vision, anonymising faces (a baseline and a privacy tool for Track B).
- [`03_object_detection_yolo_sam.ipynb`](../docs/03_object_detection_yolo_sam.ipynb): fine-tuning YOLO, evaluation with mAP, confidence thresholds, SAM (Track B).

| | |
|---|---|
| **Work in** | Teams (as formed in class) |
| **Deliverables** | One GitHub repository in the [iaac-maai GitHub organization](https://github.com/iaac-maai), one model repository in the [iaac-maai Hugging Face organization](https://huggingface.co/iaac-maai) and, for Track B with your own annotations, one dataset repository |
| **Deadline** | **Monday, 9 November 2026, 23:59 CET** |
| **How to hand in** | See [section 7 of the course README](../../README.md#7-hand-in-your-submission), and put the Hugging Face link(s) in your GitHub README |

⚠️ **On 9 November 2026 the submission links (submodules) in this repository are updated to the latest commit on your `main` branch.** That commit, and the version of your Hugging Face repositories at that time, is your submission. Anything pushed afterwards is not part of it.

---

## Table of contents

1. [Choose a track](#1-choose-a-track)
2. [Repository structure](#2-repository-structure)
3. [The notebook](#3-the-notebook)
4. [Show the model: figures](#4-show-the-model-figures)
5. [The GitHub README](#5-the-github-readme)
6. [Publish on Hugging Face](#6-publish-on-hugging-face)
7. [The model card](#7-the-model-card)
8. [The dataset card (Track B with your own annotations)](#8-the-dataset-card-track-b-with-your-own-annotations)
9. [The other files](#9-the-other-files)
10. [Submission checklist](#10-submission-checklist)
11. [Evaluation](#11-evaluation)

---

## 1. Choose a track

### 1.1 The tracks

| | **A1: MLP on tables** | **A2: image classifier** | **B: detection / segmentation** |
|---|---|---|---|
| **Input** | A row of numbers (your Day 1/2 data) | One image | One image (or video frames) |
| **Output** | A class or a number | One class per image | Boxes (or masks) with a class and a confidence for **every object** in the image |
| **What you train** | A PyTorch `nn.Sequential` MLP from scratch | A pretrained CNN (e.g. ResNet18) with a new last layer | A pretrained YOLO26 model (`yolo26n/s.pt`, or `yolo26n-seg.pt` for masks) |
| **Start from** | Notebook 01, sections 4–6 | Notebook 01 + the recipe in [section 3.2](#32-recipe-for-track-a2-fine-tuning-a-pretrained-cnn) | Notebook 03, sections 4–11 |
| **Hardware** | Any laptop CPU | A laptop CPU works for a frozen backbone; a GPU (Colab) for more | A GPU for the full model; see [1.4](#14-where-to-train) |
| **Examples** | Energy class or construction cost from design parameters; occupancy level from sensor readings | Crack / no crack in concrete; facade style; damage level after a disaster; roof or wall material | Windows and doors on facades; cracks and spalling; scaffolding, cranes, machinery; solar panels or roof types on aerial images |

💡 **Tip – which track?** A1 is the safest: you already know your data, and you can compare the network with your Day 2 model. A2 and B are closer to real deep learning (images) but need more data preparation and more computing. **Track B is the natural choice if you want to build on notebook 3**; A2 is the natural choice when one label per image is enough ("is there a crack in this photo?") and you do not need to know **where**.

### 1.2 Three questions before you start

They shape every later decision and they go in your README and model card:

1. **Who would use this model, and for what?** For example: "a facility manager screening 2,000 inspection photos for cracks before a site visit".
2. **What does a mistake cost?** A missed crack (false negative) and a false alarm (false positive) rarely cost the same. This decides your metric and, for Track B, your confidence threshold.
3. **What would be good enough?** Compared with what: your Day 2 model (A1), always predicting the most common class (A2), the pretrained model without fine-tuning, or the rule-based approach of notebook 2 (B)?

### 1.3 Which data?

| Track | Data |
|---|---|
| **A1** | The **cleaned dataset from your Day 1 or Day 2 assignment**, with the same target as on Day 2, so you can compare the network with your Day 2 model. |
| **A2** | An openly licensed image dataset with **one label per image**, for example from [Kaggle](https://www.kaggle.com/datasets), [Hugging Face](https://huggingface.co/datasets) or [Roboflow Universe](https://universe.roboflow.com), or photos you take yourselves and sort into one folder per class. |
| **B** | An openly licensed dataset **with boxes or masks** (e.g. [Roboflow Universe](https://universe.roboflow.com), which exports in YOLO format directly), **or your own photos, annotated by your team** (see [notebook 03, section 5](../docs/03_object_detection_yolo_sam.ipynb) for tools and labelling guidelines). If you annotate your own images, you **also publish the dataset** ([section 8](#8-the-dataset-card-track-b-with-your-own-annotations)). |

**How much data?**
- **A2:** at least a **few hundred images per class**. With a frozen pretrained backbone, a few hundred in total can already work.
- **B:** for fine-tuning a pretrained YOLO, **100–300 varied images and a few hundred boxes per class** is a useful start; more for rare or small objects. Variety (sites, light, distances, cameras) beats quantity.

⚠️ **Do not use the datasets from class**: house prices, energy efficiency and construction site safety (PPE). If you are unsure whether your problem is suitable, ask the course maintainers before you start.

⚠️ **Licence and privacy.** You will publish a model trained on this data (and, for Track B with own annotations, the data itself), so check **before** you start:
- The licence must allow derived models (and redistribution, if you publish the data). CC BY, CC BY-SA, CC0 and similar are fine; *NoDerivatives*, "research use only" or no licence at all are not.
- **People in images.** If your images show people, do not publish **identifiable faces**: blur them (notebook 02, section 7, shows how) or choose other images. Photos taken on a real site need the permission of the site and of the people in them.

### 1.4 Where to train

| | |
|---|---|
| **A1** | Any laptop, on the CPU: seconds to minutes. |
| **A2** | On a CPU with a **frozen** backbone (only the last layer learns): minutes to half an hour for a few thousand images. To fine-tune the whole network, use a GPU (Colab or Kaggle, free). |
| **B** | The three tiers of [notebook 03, section 6](../docs/03_object_detection_yolo_sam.ipynb): **local** (YOLO26n, a fraction of the data, 10–30 min on a CPU), **free cloud** (Colab or Kaggle T4 GPU, YOLO26s, the whole dataset) or **rented** (RunPod, Vast.ai…, a few €). Say in your README which one you used. |

⚠️ **Rented GPUs:** stop and delete the machine as soon as training ends. You pay for every hour it runs, even when idle.

---

## 2. Repository structure

Your repository must contain this structure:

```
<repo-name>/
├── data/
│   └── .gitkeep                       # keeps the folder in git
├── dataset/                           # ONLY Track B with your own annotations (section 8)
│   ├── README.md                      # the Hugging Face dataset card (committed)
│   ├── figures/                       # the figures shown in the dataset card (committed and uploaded)
│   └── images/, labels/, data.yaml    # the annotated dataset (NOT committed: uploaded to Hugging Face)
├── figures/
│   ├── 01_training_curves.png         # your exported plots
│   └── ...
├── models/
│   ├── README.md                      # the Hugging Face model card (committed)
│   ├── *.png                          # the figures shown in the model card (committed and uploaded)
│   └── model.pt                       # the trained weights (NOT committed: uploaded to Hugging Face)
├── notebook/
│   └── neural_network.ipynb           # Track A (Track B: object_detection.ipynb)
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

| Item | What goes in it |
|---|---|
| `data/` | Same rules as before: data is downloaded by the notebook, not committed. Images and label files are **never** committed. |
| `dataset/` | **Track B with your own annotations only**: exactly what you upload to Hugging Face as a dataset. The dataset card (`README.md`, and the figures it shows) is committed; the images and labels live **only** on Hugging Face. |
| `figures/` | The key plots, exported by the notebook as `.png` files, **committed** and shown in your README. |
| `models/` | Exactly what you upload to Hugging Face as a model: the model card (`README.md`), **the figures it shows** (`.png`), and the weights (`model.pt`, **only** on Hugging Face). **A1** also saves the scaler here (`scaler.joblib`, only on Hugging Face). |
| `notebook/` | Your notebook. Extra notebooks (e.g. a Colab training notebook) are allowed if you explain them in the README. |
| `.gitignore`, `LICENSE`, `requirements.txt` | See [section 9](#9-the-other-files). |
| `README.md` | The front page of your project. See [section 5](#5-the-github-readme). |

As before: your GitHub repository holds the **code that produces the results**; Hugging Face holds the **model (and the data)**. Each one links to the other.

⚠️ **Paths:** the notebook lives in `notebook/`, so the other folders are **one level up**. Use relative paths only:

```python
from pathlib import Path

DATA_DIR = Path("..") / "data"
FIG_DIR = Path("..") / "figures"
MODEL_DIR = Path("..") / "models"
DATASET_DIR = Path("..") / "dataset"   # Track B with your own annotations
```

⚠️ **Ultralytics (Track B)** writes its training runs to a global folder unless you give an **absolute** path: use `project=str((Path("..") / "runs").resolve())` in `model.train(...)`, as notebook 03 does.

---

## 3. The notebook

The notebook must **run from top to bottom without errors** (*Restart kernel → Run All*) and must be committed **with its outputs**. As before, every step needs a **markdown cell that explains what you do and why**.

💡 **Tip – training on Colab or Kaggle:** run the whole notebook there, download it **with its outputs** (*File → Download → .ipynb*) and commit that version. Keep a `TRAIN` flag, as in notebook 03, so that someone without a GPU can run the rest of the notebook with your published weights. Say in the README where the training ran.

### 3.1 Required sections

**0. Introduction and problem definition**
- Title, team members, track (A1, A2 or B) and the **use case**: the answers to the three questions of [section 1.2](#12-three-questions-before-you-start).
- The data: source, licence, size, and what one example is (a row, an image, a box).

**1. Data and splits**
- Load the data and **look at it**: a few rows (A1), a grid of images per class (A2), images with their boxes drawn and the number of boxes per class (B).
- **Split before anything else** into training, validation and test sets. The test set is used **once**, at the end. Stratify by class (A). **Split by source** (B, and A2 if relevant): frames of one video or photos of one building all go in the **same** split, or the test set is just a copy of the training set.
- **A1:** use the **same test set as your Day 2 model** (same split and `random_state`), so the comparison is fair. Scale with statistics from the training set only.

**2. Baseline**
- **A1:** your **Day 2 model**, retrained or reloaded, evaluated on the same test set.
- **A2:** the **majority class** (always predict the most common class), and if you like the pretrained network with only its last layer trained.
- **B:** the **pretrained** YOLO without fine-tuning (does it already find anything useful?), or the **rule-based** approach of notebook 02 if your objects have a clear colour or shape.

**3. The model and its training**
- **A1:** an `nn.Sequential` MLP and your training loop (notebook 01, section 4). Show the architecture with `summary(...)`.
- **A2:** a pretrained CNN with a new last layer, data augmentation, and a training loop ([section 3.2](#32-recipe-for-track-a2-fine-tuning-a-pretrained-cnn)).
- **B:** `YOLO(...).train(...)` with your training settings (model size, epochs, image size, batch) and the tier you used.
- **For every track:** the **training curves** (training and validation loss, and a metric, per epoch). Comment on them: is it still learning? Is it overfitting? When would you stop?

**4. Look inside and diagnose**
- **A1:** the inspection tools of notebook 01: `summary`, `parameter_table`, `draw_network` (before and after training), and `layer_outputs` for one example travelling through the network.
- **A2:** the confusion matrix on validation and a **gallery of misclassified images**, with the predicted class and its probability. What do the mistakes have in common?
- **B:** per-class AP, the confusion matrix, and a **gallery of failures**: missed objects (false negatives) and false alarms (false positives). What do they have in common (size, light, occlusion, an ambiguous class definition)?

**5. Tuning and regularisation**
- At least **one experiment** that changes one thing and compares the results on the **validation** set in a table. For example: early stopping, dropout or weight decay (notebook 01, section 5); the learning rate; the network size; freezing vs fine-tuning the whole network (A2); model size, image size or number of epochs (B).

**6. Test set, once**
- **A:** the main metric (accuracy, F1, MAE, R²…) and the confusion matrix (classification) or predicted vs. actual (regression), **next to the baseline**.
- **B:** mAP50 and mAP50-95 overall and **per class**, precision and recall, and the **confidence threshold** you recommend, justified with a precision/recall sweep and the cost of mistakes (notebook 03, section 9.2).

**7. Application**
Use the model for its use case:
- **A1:** predictions for a few **hand-made design options**, with the class probabilities (softmax) or the predicted value.
- **A2 / B:** run the model on **new images** (or a video) that are **not** in your dataset (your own photos, openly licensed images) and turn the result into something useful: a count, a report, a map, a ranking of the photos to check first. **B, optional:** SAM masks from your boxes, or auto-annotation (notebook 03, section 10).

**8. Save, reload and use**
- Save the weights to `models/model.pt` (**A:** `torch.save(model.state_dict(), ...)`; **A1:** also the scaler with `joblib.dump`; **B:** a copy of `best.pt`).
- **Reload** them in a new object and predict one example, as a user would (notebook 01, section 6; notebook 03, section 11).
- Export the figures for the model card into `models/` ([section 4](#4-show-the-model-figures)).

**9. Conclusions**
- The **key findings** (3–6 bullet points, with numbers): how good is it, compared with what, where does it fail?
- **Limitations and biases**: data, choices, failure cases, and what the model should **not** be used for.

### 3.2 Recipe for Track A2: fine-tuning a pretrained CNN

The class notebooks do not train a CNN, so here is a working starting point. It uses **transfer learning**: a ResNet18 pretrained on ImageNet keeps its learned filters, and only a **new last layer** learns your classes. Put the images in one folder per class: `data/images/train/crack/*.jpg`, `data/images/train/no_crack/*.jpg`, and the same for `val/` and `test/`.

```python
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

weights = models.ResNet18_Weights.DEFAULT                 # pretrained on ImageNet
eval_tf = weights.transforms()                            # resize, crop to 224 x 224, normalise as in pretraining
train_tf = transforms.Compose([                           # the same, plus random changes (augmentation)
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(eval_tf.mean, eval_tf.std),
])
train_ds = datasets.ImageFolder(DATA_DIR / "images" / "train", transform=train_tf)
val_ds = datasets.ImageFolder(DATA_DIR / "images" / "val", transform=eval_tf)
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=64)

model = models.resnet18(weights=weights)
for p in model.parameters():                              # freeze the pretrained layers...
    p.requires_grad = False
model.fc = nn.Linear(model.fc.in_features, len(train_ds.classes))   # ...and replace the last one (trainable)
model.to(DEVICE)

optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()
for epoch in range(10):
    model.train()
    for xb, yb in train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        correct = sum((model(xb.to(DEVICE)).argmax(1).cpu() == yb).sum().item() for xb, yb in val_loader)
    print(f"epoch {epoch + 1}: train loss {loss.item():.3f}, val accuracy {correct / len(val_ds):.1%}")
```

Extend it like the loop of notebook 01 (record the training and validation loss every epoch, early stopping). `train_ds.classes` gives the class names in the order of the outputs: write them in your model card. To fine-tune more, unfreeze the last block (`model.layer4`) with a 10× smaller learning rate once the new layer has learned.

💡 **Tip:** `weights.transforms()` must also be applied to every image at **prediction** time: a model fed with images prepared differently from training gives nonsense. Write this in your model card.

### 3.3 Technical requirements

- **Libraries:** PyTorch (`torch`), plus `torchvision` (A2), `ultralytics` (B), `torchinfo` (A1), and pandas / numpy / matplotlib / scikit-learn as needed.
- A fixed **seed** everywhere randomness is involved (`torch.manual_seed`, `random_state`, `seed=` in Ultralytics).
- At least **4 figures** exported to `figures/` ([section 4](#4-show-the-model-figures)).
- **No credentials** in the notebook: no Kaggle keys, **no Hugging Face tokens**, no Roboflow API keys.

---

## 4. Show the model: figures

A model card full of numbers asks the reader to **trust** you. A model card with figures lets them **see** for themselves, in a few seconds, what the model does and how good it is, **before** they download it. That is why figures are a **requirement** of this assignment, not decoration:

- Your **GitHub README** must embed **at least one figure, and preferably several**.
- Your **model card** must embed **at least one figure, and preferably several**.
- Together they must show the three things below. Choose the figures that tell your model's story best.

| Show… | Track A1 | Track A2 | Track B |
|---|---|---|---|
| **How it was trained** (the process) | Training and validation loss/accuracy per epoch; `draw_network` before and after training | Training and validation loss/accuracy per epoch | Training curves from `results.csv` (box/class loss, mAP50 per epoch) |
| **How well it performs** | Confusion matrix (classification) or predicted vs. actual (regression); a bar chart **network vs. Day 2 model vs. baseline** | Confusion matrix; per-class precision/recall; network vs. baseline | Per-class AP bar chart; confusion matrix; precision/recall against the confidence threshold |
| **What it does** | One example travelling through the network (`layer_outputs`), or the predictions for a few design options | A grid of test images with the true class, the predicted class and the probability, **including mistakes** | Test images with the **true boxes and the predicted boxes** side by side; a gallery of failures; your application (a video frame, a report) |

The [course model card](https://huggingface.co/iaac-maai/day4-example-ppe-detector) (source: [`Day_4/docs/models/ppe_detector/README.md`](../docs/models/ppe_detector/README.md)) is a worked example: training curves, a confusion matrix, predictions next to the true boxes, a threshold sweep and frames from a site video, each with a short reading.

**Rules for good figures:**
- Every figure has a **title, axis labels and units**, and readable text.
- Every figure in a card or README has a **one- or two-sentence reading** below it: what should the reader notice?
- **Show mistakes, not only successes.** A gallery of perfect predictions is an advertisement, not an evaluation.
- Images of people: faces blurred ([section 1.3](#13-which-data)).

**How to embed them:**
- **README:** `![Training curves](figures/01_training_curves.png)`.
- **Model card:** copy the figures into `models/` (the notebook can save them there directly), upload them with the model, and embed them by their **Hugging Face address**, so that they render both on Hugging Face and on GitHub:

```markdown
![Training curves](https://huggingface.co/iaac-maai/<model-repo>/resolve/main/training_curves.png)
```

---

## 5. The GitHub README

Someone who has never seen your project should understand it and be able to run it **in under five minutes**. It must contain:

1. **Title**, **team members** and **track**.
2. **Use case**: who uses the model, for what, and what a mistake costs (2–4 sentences).
3. **On Hugging Face**: the link to your **model** (and your **dataset**, Track B with own annotations), at the top, impossible to miss.
4. **Dataset**: name, links, short description, size, licence.
5. **Repository structure**: the folder tree with one line per item.
6. **How to run**: environment, `requirements.txt`, data, where the training ran (laptop, Colab, Kaggle, rented GPU) and how long it took.
7. **Approach**: architecture, training settings, baseline, the experiments of section 5 of the notebook.
8. **Results**: a table with the test metrics **next to the baseline**, and **several figures** from `figures/`: at least one showing the training, one showing the performance and one showing what the model does ([section 4](#4-show-the-model-figures)).
9. **Limitations**.
10. **AI usage disclosure**: same rules as Days 1–3, see [below](#ai-usage-disclosure).
11. **License**.

### AI usage disclosure

Using AI tools is allowed, but it must be disclosed. As before, this section is also a way to **share with the class how to use agentic tools effectively**, so be concrete:

- **Which tools** did you use (e.g. ChatGPT, Claude, Claude Code, GitHub Copilot, Cursor)? Did **agents** edit files or run code by themselves?
- **Why**: what did you expect the tool to help with?
- **Where**: which parts (data preparation, annotation, training code, debugging, figures, model card, README)?
- **How**: your workflow. What did you ask for (an example prompt helps), how did you check the output, and what did you correct or reject?
- **Lessons learned**: what worked, what did not, what would you do differently?

If you did not use AI tools, write one sentence saying so.

⚠️ You are responsible for everything in your submission, including what an AI tool produced. **The reading of your results and failure cases must be your team's own**, and you must be able to defend it.

A starting template:

````markdown
# <Project title>

**Team:** Name Surname, Name Surname, Name Surname
**Course:** AI Methods I – Day 4: Neural networks and computer vision (IAAC MaAI 2026/27)
**Track:** A1 (MLP on tables) / A2 (image classifier) / B (detection / segmentation)
**Model on Hugging Face:** https://huggingface.co/iaac-maai/<model-repo>
**Dataset on Hugging Face:** https://huggingface.co/datasets/iaac-maai/<dataset-repo>   <!-- Track B with own annotations only -->

![What the model does](figures/05_predictions.png)
<!-- One figure at the top that shows the model at work: predictions on test images, or one example through the network -->

## Use case
...

## Dataset
| | |
|---|---|
| **Name** | ... |
| **Source** | [link](...) |
| **Size** | ... rows / images / boxes per class |
| **Licence** | ... |

## Repository structure
```
├── data/        # downloaded data (not committed)
├── figures/     # exported plots
├── models/      # model card (README.md) + its figures; model.pt lives on Hugging Face
├── notebook/    # neural_network.ipynb / object_detection.ipynb
...
```

## How to run
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
Open `notebook/<notebook>.ipynb`, select the `.venv` kernel and run all cells.
Training ran on ... (laptop CPU / Colab T4 / ...) and took ... minutes. Without a GPU, set `TRAIN = False` to use the published weights.

## Approach
- **Architecture:** ... (... parameters)
- **Training:** ... epochs, batch ..., learning rate ..., augmentation ...
- **Baseline:** ...
- **Experiments:** ...

## Results
| Model | <metric> (test) |
|---|---|
| Baseline | ... |
| Our network | ... |

![Training curves](figures/01_training_curves.png)
*...one or two sentences: what to notice...*

![Confusion matrix](figures/03_confusion_matrix.png)
*...*

![Failure cases](figures/04_failures.png)
*...*

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
Code released under the [MIT License](LICENSE). The model is released under ... (see the model card). The data is subject to its own licence (see above).
````

---

## 6. Publish on Hugging Face

You publish exactly as on Days 2 and 3. If you have not set up Hugging Face yet (account, membership of the **iaac-maai** organization, the `hf` command-line tool, an access token and `hf auth login`), follow **[section 5.1 of the Day 2 brief](../../Day_2/submissions/README.md#51-one-time-setup)** first.

⚠️ **Never put the token in your notebook, your code or your repository.**

### 6.1 The model

**Name** your model repository with this pattern, all lowercase, with hyphens:

```
iaac-maai/day4-<team>-<what-it-does>
```

For example `iaac-maai/day4-team3-energy-class-mlp`, `iaac-maai/day4-team5-crack-classifier` or `iaac-maai/day4-team2-window-detector`.

**Create and upload** from the **root of your GitHub repository**, after the notebook has saved the weights and the figures into `models/` and you have written `models/README.md` (the model card, [section 7](#7-the-model-card)):

```bash
# 1. Create the model repository inside the organization
hf repos create iaac-maai/day4-team5-crack-classifier --type model --exist-ok

# 2. Upload the contents of models/ (model.pt, the figures and README.md) to the root of the repository
hf upload iaac-maai/day4-team5-crack-classifier models . --commit-message "Upload weights, figures and model card"
```

Open the model page and check that the card **renders with all its figures**.

### 6.2 The dataset (Track B with your own annotations)

Export the dataset in **YOLO format** into `dataset/` (`images/` and `labels/` with `train/`, `val/` and `test/` subfolders, and `data.yaml`), write the dataset card ([section 8](#8-the-dataset-card-track-b-with-your-own-annotations)), and upload it as a **dataset** repository (note `--type dataset` in **both** commands):

```bash
hf repos create iaac-maai/day4-team2-facade-windows --type dataset --exist-ok
hf upload iaac-maai/day4-team2-facade-windows dataset . --type dataset --commit-message "Upload annotated images and dataset card"
```

⚠️ **Watch out – tell the Data Viewer where the data is.** Without instructions, Hugging Face guesses which files are the dataset, and it often guesses wrong: the **figures of your dataset card** (`.png`) get shown as if they were dataset images, or a single CSV is hidden behind a few plots. Keep the card's figures in a `figures/` subfolder and add a `configs` block to the YAML header of the dataset card that points at the real data:

```yaml
configs:
  - config_name: default
    data_files:
      - split: train
        path: images/train/*
      - split: validation
        path: images/val/*
      - split: test
        path: images/test/*
```

(For a tabular dataset, use one entry instead: `- split: train` with `path: your_file.csv`.) After uploading, open the **Dataset Viewer** on the dataset page and check that it shows your images (or your table's columns), not the card's figures. It can take a few minutes to refresh.

### 6.3 Test it like a stranger

Download your model into a **fresh** session (a new notebook, or Colab) and use it on one example, with nothing but the model card to guide you. If this fails, fix it before the deadline.

⚠️ **Tracks A1 and A2: a `state_dict` contains only the numbers, not the architecture.** To load it, the user must first **build the same network** in code. Your model card must therefore contain the **exact code that builds the model** (A1: your `nn.Sequential`; A2: the torchvision model with its new last layer), the class names in order, and the preprocessing (A1: the scaler; A2: the transforms).

**A1**: rebuild, load the weights and the scaler, predict:

```python
import joblib
import pandas as pd
import torch
from torch import nn
from huggingface_hub import hf_hub_download

REPO = "iaac-maai/day4-team3-energy-class-mlp"
model = nn.Sequential(                                     # EXACTLY the architecture you trained,
    nn.Linear(8, 32), nn.ReLU(), nn.Dropout(0.2),          # including Dropout layers: they shift the layer
    nn.Linear(32, 32), nn.ReLU(), nn.Dropout(0.2),         # numbers in the state_dict, and loading fails
    nn.Linear(32, 3),                                      # without them
)
model.load_state_dict(torch.load(hf_hub_download(REPO, "model.pt"), map_location="cpu"))
model.eval()
scaler = joblib.load(hf_hub_download(REPO, "scaler.joblib"))

example = pd.DataFrame([{ ... }])                          # one row with the same columns, in the same order
x = torch.tensor(scaler.transform(example), dtype=torch.float32)
with torch.no_grad():
    print(torch.softmax(model(x), dim=1))                  # probability of each class
```

**A2**: rebuild, load the weights, predict one image with the same transforms:

```python
import torch
from torch import nn
from torchvision import models
from PIL import Image
from huggingface_hub import hf_hub_download

CLASSES = ["crack", "no_crack"]                            # train_ds.classes, in this order
model = models.resnet18()
model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
model.load_state_dict(torch.load(hf_hub_download("iaac-maai/day4-team5-crack-classifier", "model.pt"), map_location="cpu"))
model.eval()

transform = models.ResNet18_Weights.DEFAULT.transforms()   # the same preprocessing as in training
image = Image.open("wall.jpg").convert("RGB")
with torch.no_grad():
    probabilities = torch.softmax(model(transform(image).unsqueeze(0)), dim=1)[0]
print(dict(zip(CLASSES, probabilities.tolist())))
```

**B**: Ultralytics weights contain the architecture and the class names, so loading is one line:

```python
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

model = YOLO(hf_hub_download("iaac-maai/day4-team2-window-detector", "model.pt"))
result = model.predict("facade.jpg", conf=0.25)[0]          # your recommended threshold
print(len(result.boxes), "objects:", [model.names[int(c)] for c in result.boxes.cls])
result.save("facade_detections.jpg")
```

---

## 7. The model card

The model card is the `README.md` of your Hugging Face model repository (written in `models/README.md`, committed to GitHub, uploaded to Hugging Face). As on Days 2 and 3, it is the **user manual of your model**. Today it must also **show** the model: a reader should see what it does and how well, **before** using it.

💡 **Reference:** the course model [iaac-maai/day4-example-ppe-detector](https://huggingface.co/iaac-maai/day4-example-ppe-detector) (source: [`Day_4/docs/models/ppe_detector/README.md`](../docs/models/ppe_detector/README.md)) is a complete, filled-in model card for Track B. Tracks A1 and A2 follow the same structure with their own metrics and figures.

### 7.1 What it must contain

| Section | What to write |
|---|---|
| **Metadata header** | Licence, library (`pytorch` for A, `ultralytics` for B), pipeline tag, tags, and `datasets:` if your data is on Hugging Face |
| **Model summary** | One paragraph: what it does, from what input, for whom, with which architecture |
| **Model at a glance** | **Right after the summary: at least one figure, preferably several**, that show what the model does and how well it works ([section 4](#4-show-the-model-figures)), each with a one- or two-sentence reading |
| **Intended use** | Who should use it and for what. **Out-of-scope uses**: where it must *not* be used (other building types, regions, image types, value ranges; any decision about individual people) |
| **Model details** | Architecture and **number of parameters**, base model and what it was pretrained on (A2, B), input (columns or image size), output (classes or values), files, library versions |
| **Classes / input features** | **A1:** a table of every input column (type, unit, preprocessing). **A2 / B:** a table of the classes, with a **definition** of each (what counts, what does not) |
| **Training data** | Dataset name, **source link**, licence, size (rows / images / boxes per class), splits, coverage, cleaning or annotation |
| **Training procedure** | Epochs, batch size, optimiser and learning rate, augmentation, regularisation, **hardware and training time**, how the final weights were selected; **the training curves** |
| **Evaluation** | Test metrics **next to the baseline**; per-class scores; the confusion matrix; **where it fails**, with examples. **B:** the recommended **confidence threshold** and why |
| **Limitations and biases** | What the data does not represent, known failure cases, sensitivity to image conditions. Be honest: this is the most important section for a user |
| **Ethics and privacy** | **Required if your images show people**: consent, faces, what the model must not be used for. Optional otherwise |
| **How to use** | A short, **working** snippet ([section 6.3](#63-test-it-like-a-stranger)): for A, including the code that builds the architecture and the preprocessing |
| **Reproducibility** | Link to the **GitHub repository**, the notebook, and the `requirements.txt` / library versions |
| **Security note** | `.pt` (and `.joblib`) files are pickles: they can run code when loaded. Users should only load models from sources they trust |
| **Authors, course and licence** | Team members, "AI Methods I, IAAC MaAI 2026/27", licence of the model and of the data |

### 7.2 Template

Copy this into `models/README.md` and fill in every `...`. Delete the comments (`<!-- -->`) and the lines of the other tracks when you are done.

````markdown
---
license: mit                            # Track B: agpl-3.0 (the licence of the Ultralytics models)
library_name: pytorch                   # Track B: ultralytics
pipeline_tag: image-classification      # A1: tabular-classification / tabular-regression; B: object-detection / image-segmentation
tags:
  - pytorch
  - built-environment
  - iaac-maai
  - ai-methods-1
datasets:                               # only if your data is on Hugging Face
  - iaac-maai/<dataset-repo>
metrics:
  - accuracy                            # A2 e.g. accuracy / f1; B: map50
---

# <Model name>: <what it does>

<!-- One paragraph: what it does, from what input, for whom, with which architecture. -->
This model ... It is a <architecture> (... parameters) <trained from scratch / fine-tuned from ... pretrained on ...> on ... It was built as a course assignment for AI Methods I (IAAC MaAI 2026/27).

- **GitHub repository (full pipeline):** https://github.com/iaac-maai/<repo-name>
- **Team:** Name Surname, Name Surname, Name Surname

## Model at a glance

![What the model does](https://huggingface.co/iaac-maai/<model-repo>/resolve/main/predictions.png)

*<!-- One or two sentences: what the reader sees, including a mistake. -->*

| Model | <main metric> on the test set |
|---|---|
| Baseline: ... | ... |
| **This model** | **...** |

## Intended use

- **Intended users and use:** ...
- **Out-of-scope uses:** ...

## Model details

| | |
|---|---|
| **Task** | e.g. image classification / object detection / tabular classification |
| **Architecture** | e.g. ResNet18 with a new last layer; YOLO26s; MLP 8 → 32 → 32 → 3 with ReLU |
| **Parameters** | ... (... trainable) |
| **Base model** | e.g. ResNet18 pretrained on ImageNet / `yolo26s.pt` pretrained on COCO / none (trained from scratch) |
| **Input** | e.g. one RGB image, resized to 224 × 224 and normalised with ImageNet statistics / 8 numeric columns, standardised |
| **Output** | e.g. one probability per class / boxes with class and confidence |
| **Files** | `model.pt` (weights) <!-- A1: + scaler.joblib --> |
| **Library versions** | torch ..., torchvision ... / ultralytics ..., Python ... |

### Classes <!-- A1: "Input features", with a table of the columns instead -->

| ID | Class | Definition |
|---|---|---|
| 0 | ... | ... (what counts, what does not) |
| 1 | ... | ... |

## Training data

| | |
|---|---|
| **Dataset** | ... |
| **Source** | [link](...) |
| **Licence** | ... |
| **Size** | ... images (train ... / validation ... / test ...); ... boxes per class |
| **Coverage** | where, when, which kind of images or buildings |

## Training procedure

- **Settings:** ... epochs, batch ..., optimiser ..., learning rate ..., image size ...
- **Augmentation and regularisation:** ...
- **Hardware and time:** ... (e.g. Colab T4 GPU, 25 minutes)
- **Model selection:** e.g. the weights with the lowest validation loss; the test set was used once, at the end.

![Training curves](https://huggingface.co/iaac-maai/<model-repo>/resolve/main/training_curves.png)

*<!-- What to notice: still learning? overfitting? -->*

## Evaluation

| Metric | Baseline | This model |
|---|---|---|
| ... | ... | ... |

**Per class:**

| Class | Precision | Recall | <!-- A: F1; B: AP50 --> |
|---|---|---|---|
| ... | ... | ... | ... |

![Confusion matrix](https://huggingface.co/iaac-maai/<model-repo>/resolve/main/confusion_matrix.png)

**Where it fails:** ...

![Failure cases](https://huggingface.co/iaac-maai/<model-repo>/resolve/main/failures.png)

<!-- Track B: a "Choosing a confidence threshold" subsection with the sweep figure and your recommendation. -->

## Limitations and biases

- ...

## Ethics and privacy

<!-- Required if the images show people. -->
...

## How to use

```python
# The working snippet of section 6.3 of the brief, adapted to your model
```

⚠️ `.pt` files are Python pickles and can execute code when loaded: only load models from sources you trust. Use the library versions listed above.

## Reproducibility

The full pipeline is in the GitHub repository: https://github.com/iaac-maai/<repo-name>. Run `notebook/<notebook>.ipynb` with the packages in `requirements.txt`.

## Authors and licence

Name Surname, Name Surname, Name Surname. Course assignment for **AI Methods I: Machine Learning for the Built Environment**, IAAC MaAI 2026/27.

Model released under ... The data is subject to its own licence: ...
````

---

## 8. The dataset card (Track B with your own annotations)

If your team annotated its own images, the annotated dataset is a valuable result in itself: publish it, with a **dataset card** (`dataset/README.md`, committed to GitHub, uploaded to Hugging Face). Use the structure of the [Day 3 dataset card](../../Day_3/submissions/README.md#7-the-dataset-card), with these image-specific sections instead of the cluster ones:

| Section | What to write |
|---|---|
| **Metadata header** | Licence (e.g. `cc-by-4.0` for your own photos), `task_categories: [object-detection]` (or `image-segmentation`), size category, tags, and a **`configs` block** that points the Dataset Viewer at your images ([section 6.2](#62-the-dataset-track-b-with-your-own-annotations)) |
| **Summary** | What the images show, where and when they were taken, how many images and boxes |
| **Source and collection** | Who took the photos (or their original source and licence), with which camera, and **permissions** (site, people) |
| **Annotation** | The tool (makesense.ai, Roboflow, CVAT…), who annotated, and the **class definitions and labelling rules** you followed (what counts as a "window"? Are half-hidden ones labelled?) |
| **Contents** | The folder structure (YOLO format), the splits and how they were made (by building, by site, by date), the **number of images and boxes per class and split** |
| **Figures** | **At least one**: example images **with their boxes drawn**, and a bar chart of the boxes per class |
| **Limitations and biases** | What the images do not cover (light, seasons, building types, viewpoints), annotation mistakes, class imbalance |
| **Privacy** | Faces and number plates blurred? People informed? |
| **How to load it** | How to download it and train on it (`hf download ... --repo-type dataset`, then `YOLO(...).train(data="data.yaml")`) |
| **Related, authors and licence** | Links to the GitHub repository and the model, team, course, licence |

---

## 9. The other files

### `.gitignore`

The Day 3 version, plus model weights, training runs and images:

```gitignore
# virtual environments
.venv/
venv/

# Jupyter
.ipynb_checkpoints/

# data: downloaded by the notebook, not committed
data/*
!data/.gitkeep

# model weights and published data: on Hugging Face, not on GitHub
*.pt
*.onnx
models/*.joblib
dataset/images/
dataset/labels/
test_download/

# training outputs (Ultralytics) and videos
runs/
datasets/
*.mp4
*.webm

# OS files
.DS_Store
Thumbs.db

# secrets
.env
kaggle.json
```

⚠️ Do **not** ignore `figures/`, `*.png`, `models/README.md`, the figures in `models/`, or `dataset/README.md`: they must be on GitHub.

### `requirements.txt`

Only the packages your notebook imports, **with versions** (the course versions are in the [course `requirements.txt`](../../requirements.txt)). For example:

```text
torch==2.14.1
torchvision==0.29.1        # Track A2
torchinfo==1.8.0           # Track A1
ultralytics==8.4.190       # Track B
pandas==3.0.6
numpy==2.5.3
matplotlib==3.11.2
scikit-learn==1.9.1
joblib==1.6.0
opencv-python==4.14.0.94   # if you use OpenCV
huggingface_hub==2.2.0
```

Pin **torch** and **ultralytics** exactly: your model card refers to these versions.

### `LICENSE`

**MIT License** for your code, as before. For the **model**:
- **Track A:** MIT, as your code.
- **Track B:** **AGPL-3.0**, because it is fine-tuned from Ultralytics YOLO models, which are released under that licence. Use `license: agpl-3.0` in the model card, as the course model does.

The data keeps **its own licence**: mention it in the README and the cards.

---

## 10. Submission checklist

**GitHub repository**
- [ ] The repository is inside the **iaac-maai** GitHub organization
- [ ] It has `data/`, `figures/`, `models/` (with the model card and its figures), `notebook/`, `.gitignore`, `LICENSE`, `README.md` and `requirements.txt` (and `dataset/` with the dataset card, Track B with own annotations)
- [ ] No virtual environment, data, images, `.pt` / `.joblib` files, training runs or credentials are committed
- [ ] The final version is pushed to `main` **before 9 November 2026**

**Notebook**
- [ ] It runs from top to bottom without errors and is committed **with outputs** (training on Colab/Kaggle: the downloaded notebook with its outputs, and a `TRAIN` flag)
- [ ] It states the track, the use case and what a mistake costs
- [ ] The data is split **before** anything else, by source where needed, and the test set is used **once**
- [ ] It compares the network with a **baseline** (A1: your Day 2 model on the same test set)
- [ ] It shows and comments the **training curves**
- [ ] It looks inside or diagnoses the model (A1: the notebook 01 tools; A2/B: a gallery of mistakes)
- [ ] It contains at least **one tuning or regularisation experiment**, compared on validation
- [ ] **B:** per-class AP and a justified confidence threshold
- [ ] It applies the model to its use case (design options, or new images / video)
- [ ] It saves the weights, reloads them and predicts one example
- [ ] It exports at least 4 figures to `figures/` and the model card figures to `models/`

**Figures ([section 4](#4-show-the-model-figures))**
- [ ] The **README** embeds several figures: how the model was trained, how well it performs, and what it does
- [ ] The **model card** has a *Model at a glance* section with at least one figure, and its figures **render on Hugging Face**
- [ ] Every figure has a title, labelled axes and a short reading; the figures show **mistakes**, not only successes

**Hugging Face**
- [ ] The model repository is inside the **iaac-maai** organization and named `day4-<team>-<what-it-does>`
- [ ] It contains `model.pt` (A1: and `scaler.joblib`), the figures and a complete **model card** (every section of [7.1](#71-what-it-must-contain))
- [ ] **A:** the card contains the code that builds the architecture, the class order and the preprocessing
- [ ] It loads and predicts in a fresh session ([section 6.3](#63-test-it-like-a-stranger))
- [ ] The model card links to the GitHub repository, and the GitHub README links to the model
- [ ] **B with own annotations:** the dataset is published as `iaac-maai/day4-<team>-<dataset-name>` with a dataset card ([section 8](#8-the-dataset-card-track-b-with-your-own-annotations)), and published images contain no identifiable faces
- [ ] **B with own annotations:** the dataset card has a `configs` block, and the **Dataset Viewer** shows your images, not the card's figures

**README**
- [ ] Team, track, use case, Hugging Face links, dataset, structure, how to run (including where the training ran), approach, results (metrics next to the baseline + several figures), limitations, AI usage disclosure, licence

---

## 11. Evaluation

The assignment is graded following the course evaluation criteria (see the syllabus). In particular:

| Criterion | What we look for |
|---|---|
| **Completeness** | All required files, sections and steps are present; the notebook runs; the model is published, loads and predicts |
| **Methodology** | Clean splits (by source where needed) and a test set used once; a fair **baseline**; training curves read correctly; at least one well-designed experiment; for B, a justified confidence threshold |
| **Understanding** | You can explain what the network learned and **where and why it fails**: the inspection tools (A1) or the failure gallery (A2, B) are read, not just shown |
| **Showing the model** | The README and the model card let a reader **see** how the model was trained, how well it performs and what it does **before using it**: several clear, labelled figures, each with a reading, including the mistakes |
| **Critical thinking** | Honest comparison with the baseline (a network is not automatically better); limitations and biases; licence, privacy and ethics checked, especially for images of people |
| **Communication** | Clear notebook, README and **model card** that a stranger can follow; a working *How to use* snippet; an honest and useful AI usage disclosure |
| **Relevance** | The problem and the model are meaningful for architecture and the built environment |
| **Repository and publication conventions** | Clean structure, sensible `.gitignore`, pinned `requirements.txt`, licences, correct Hugging Face naming, metadata and repository types |

A modest model that is honestly evaluated and clearly **shown** is worth more than a high score that nobody can see or check. Your model card is how other people will meet your model: make it something they can trust at a glance.
