---
license: agpl-3.0
library_name: ultralytics
pipeline_tag: object-detection
tags:
  - ultralytics
  - yolo
  - yolo26
  - object-detection
  - construction
  - safety
  - ppe
  - built-environment
  - iaac-maai
  - ai-methods-1
metrics:
  - map50
  - map
---

# Construction site safety detector (YOLO26s): helmets, vests, masks, people and machinery

This model detects **personal protective equipment (PPE)** and related objects on construction sites: hard hats, safety vests, face masks, people, safety cones, machinery and vehicles, and also people **without** a hard hat, vest or mask. It is a **YOLO26s** detector, pretrained by Ultralytics on COCO and fine-tuned on about 2,600 labelled construction-site images.

It is the **worked example** for Day 4 of **AI Methods I: Machine Learning for the Built Environment** (IAAC MaAI 2026/27). Students without a GPU download it in the course notebook instead of training their own.

- **GitHub repository (full pipeline):** https://github.com/iaac-maai/ai-methods-1-2026 (notebook `Day_4/docs/03_object_detection_yolo_sam.ipynb`)
- **Authors:** AI Methods I course team, IAAC

## Intended use

- **Intended users and use:** students, architects and site teams who want to **learn** how object detection works, or to make **rough, aggregated** analyses of site photos and video, for example how often hard hats or vests appear in a set of photos. It is a **teaching model**.
- **Out-of-scope uses:**
  - Any decision about an **individual** worker (discipline, payment, access to the site).
  - Any **safety-critical** use without a human checking every alert: the model misses about a third of the people without a hard hat (see *Evaluation*).
  - Monitoring workers who have not been **informed** (see *Ethics and privacy*).
  - Very different images from the training data: drones or top-down views, night or thermal images, very distant people.

## Model details

| | |
|---|---|
| **Task** | Object detection (boxes, classes and confidences) |
| **Architecture** | Ultralytics YOLO26s: 9.5 million parameters, 20.8 GFLOPs |
| **Base model** | `yolo26s.pt`, pretrained on COCO (80 everyday classes) |
| **Input** | An image of any size: it is resized to 640 × 640 pixels internally |
| **Output** | For each object: a box (x1, y1, x2, y2 in pixels), one of the 10 classes below, and a confidence between 0 and 1 |
| **Files** | `model.pt` (PyTorch weights for Ultralytics, 20 MB); `results.csv` (the training log: losses and validation metrics for every epoch) |
| **Speed** | About 0.1 s per image on a laptop CPU (Intel i7-12700H); about 35 ms on a laptop RTX 4050 GPU |
| **Library versions** | ultralytics 8.4.190, torch 2.14.1, Python 3.12 |

### Classes

| ID | Class | Meaning |
|---|---|---|
| 0 | `Hardhat` | A safety helmet worn on a head |
| 1 | `Mask` | A face mask worn on a face |
| 2 | `NO-Hardhat` | A head **without** a safety helmet |
| 3 | `NO-Mask` | A face **without** a mask |
| 4 | `NO-Safety Vest` | An upper body **without** a hi-vis vest |
| 5 | `Person` | A person |
| 6 | `Safety Cone` | A traffic or safety cone |
| 7 | `Safety Vest` | A hi-vis safety vest worn by a person |
| 8 | `machinery` | Construction machinery (excavators, loaders, telehandlers…) |
| 9 | `vehicle` | Cars, vans and trucks |

## Training data

| | |
|---|---|
| **Dataset** | Construction Site Safety Image Dataset |
| **Source** | [Roboflow Universe](https://universe.roboflow.com/roboflow-universe-projects/construction-site-safety), via [Kaggle](https://www.kaggle.com/datasets/snehilsanyal/construction-site-safety-image-dataset-roboflow) |
| **Licence** | CC BY 4.0 |
| **Images** | 2,605 training, 114 validation, 82 test (640 × 640 pixels) |
| **Content** | Site photos, frames from site videos, close-up photos of people with and without face masks, and some background images with no objects |

**Notes on the data:**
- The training set contains **augmented copies** made by the dataset author (flips, crops, small rotations, brightness and blur changes), so it has fewer distinct photos than its number of files suggests.
- The classes are **imbalanced**: 9,532 `Person` boxes in the training set, but only about 1,550–1,650 `vehicle` and `Mask` boxes.
- The validation and test sets are **small**, so the scores below are uncertain by a few points.

## Training procedure

- **Fine-tuning** of `yolo26s.pt` for **40 epochs**, image size 640, batch 16, Ultralytics default optimiser and learning-rate schedule, mixed precision, `seed=42`.
- **Augmentation during training** (Ultralytics defaults): mosaic (switched off for the last 10 epochs), random scaling, translation, horizontal flips and colour changes.
- **Hardware and time:** one NVIDIA RTX 4050 Laptop GPU (6 GB), **33 minutes**.
- **Model selection:** the weights of the epoch with the best validation score (`best.pt`); the test set was used **once**, at the end.

![Training curves](https://huggingface.co/iaac-maai/day4-example-ppe-detector/resolve/main/training_curves.png)

The validation mAP50 was already 0.55 after one epoch (thanks to the pretrained weights), 0.77 after 11 epochs and 0.85 after 40, still rising slowly: longer training would gain a little more.

## Evaluation

**Overall:**

| Split | mAP50 | mAP50-95 | Precision | Recall |
|---|---|---|---|---|
| Validation (114 images) | 0.844 | 0.550 | 0.913 | 0.759 |
| **Test (82 images, used once)** | **0.780** | **0.500** | **0.877** | **0.732** |

For comparison, the dataset author reports a validation mAP50 of 0.81 (mAP50-95 0.51) for a YOLOv8n trained for 100 epochs.

**Per class, on the test set** (sorted from weakest to strongest):

| Class | Boxes in test set | Precision | Recall | AP50 | AP50-95 |
|---|---|---|---|---|---|
| Safety Cone | 92 | 0.77 | 0.48 | 0.52 | 0.24 |
| NO-Hardhat | 41 | 0.79 | 0.59 | 0.56 | 0.34 |
| Mask | 28 | 0.95 | 0.75 | 0.75 | 0.55 |
| vehicle | 41 | 0.82 | 0.68 | 0.76 | 0.48 |
| NO-Mask | 79 | 0.88 | 0.70 | 0.80 | 0.35 |
| NO-Safety Vest | 90 | 0.97 | 0.77 | 0.83 | 0.53 |
| Person | 174 | 0.87 | 0.80 | 0.86 | 0.59 |
| machinery | 44 | 0.88 | 0.84 | 0.88 | 0.69 |
| Safety Vest | 61 | 0.87 | 0.82 | 0.89 | 0.62 |
| Hardhat | 110 | 0.98 | 0.90 | 0.96 | 0.61 |

![Confusion matrix on the test set](https://huggingface.co/iaac-maai/day4-example-ppe-detector/resolve/main/confusion_matrix.png)

*How to read it:* each column is a true class and sums to 1. The diagonal is the share of each class found with the right label (at confidence ≥ 0.25), and the classes are **sorted from best to worst** along it, from `Hardhat` (0.96) to `Safety Cone` (0.67). The bottom row, "background", is the share of objects the model **missed** entirely; the right-hand column shows what its false detections were. This order differs slightly from the AP50 table above, because AP50 also accounts for false alarms and for all confidence levels.

**Where it performs worst:** the model rarely confuses one class with another; it mostly **misses objects**. In the plot, the bottom row grows steadily towards the right: the weakest classes are weak because they are missed. On the test set it misses **33% of the safety cones** (often small and far away) and **24% of the heads without a hard hat**, and takes another 7% of bare heads for hard hats. Its false detections are mostly `Person` (26% of them) and `Safety Cone` (16%).

![Predictions on test images](https://huggingface.co/iaac-maai/day4-example-ppe-detector/resolve/main/predictions.png)

*Top: the true boxes. Bottom: the model's predictions (confidence ≥ 0.25).*

### Choosing a confidence threshold

The confidence threshold trades precision for recall. Measured on the test set (a prediction counts as correct if its IoU with a true box of the same class is at least 0.5):

| Threshold | NO-Safety Vest: precision / recall | NO-Hardhat: precision / recall |
|---|---|---|
| 0.05 | 0.66 / 0.82 | 0.68 / 0.66 |
| 0.20 | **0.90 / 0.81** | 0.80 / 0.59 |
| 0.30 | 0.93 / 0.78 | 0.79 / 0.56 |
| 0.50 | 0.95 / 0.70 | 0.82 / 0.56 |
| 0.70 | 1.00 / 0.58 | 0.82 / 0.44 |

![Precision and recall against the confidence threshold](https://huggingface.co/iaac-maai/day4-example-ppe-detector/resolve/main/threshold_sweep.png)

**Recommendation:** a threshold around **0.2–0.3** for safety checks, where missing a problem is worse than a false alarm. For `NO-Hardhat`, no threshold gives good recall (at most 0.66): this class needs more training data.

### On video

We also ran the model on a 2025 construction-site video from London, which is **not** in the training data (confidence threshold 0.3):

![Detections on frames of a site video](https://huggingface.co/iaac-maai/day4-example-ppe-detector/resolve/main/video_frames.jpg)

*Frames from "Building construction Moira Close Broadwater Farm Haringey 2025 21" by [Acabashi](https://commons.wikimedia.org/wiki/File:Building_construction_Moira_Close_Broadwater_Farm_Haringey_2025_21.webm), CC BY-SA 4.0, with the model's detections drawn on them.*

- **Machinery seen from any side (top):** the telehandler is detected in practically every frame, from the front (60 s) and after it has turned round (97 s). The model has learned what machinery looks like in general, not this particular machine from one side.
- **False alarms remain (bottom left):** at 100 s a tall, wrapped insulation stack is taken for a `Person` (confidence 0.39), next to the correct detections of the telehandler and of a worker with his vest.
- **Not only yellow vests (bottom right):** the two workers in **orange** hi-vis on the left get `Safety Vest` boxes, as well as the workers in yellow, although trees hide much of the scene.

**Tracking needs care.** Ultralytics can link detections across frames with the **ByteTrack** tracker (`model.track`), giving every person an ID. On a 38-second clip with a handful of workers, the model detects 3.7 people per frame on average, but the tracker hands out **52 different IDs**:

![People per frame versus distinct tracker IDs](https://huggingface.co/iaac-maai/day4-example-ppe-detector/resolve/main/tracking_ids.png)

Every time a worker disappears behind the leaves or another worker, the tracker loses him and gives him a **new ID** when he reappears (processing only every second frame, as here, makes this worse). Never report raw tracker IDs as a count of people.

## Limitations and biases

- **Missed safety problems:** about a third of the people without a hard hat are not detected. Never treat "no alert" as "no problem".
- **Training data:** mostly daytime photos at eye level, from a mix of countries and sources, including many close-up photos (selfies with masks). Drone views, night scenes, rain, very crowded sites and very distant workers are under-represented.
- **Small objects:** cones and distant people are detected less well; higher image resolutions help but are slower.
- **Class definitions:** "NO-Hardhat" and similar classes describe an **absence**, which is ambiguous when a head is partly hidden or seen from behind.
- **Small test set:** 82 images; the scores are uncertain by a few points.

## Ethics and privacy

This model looks at **people at work**. Before using it on real sites:
- **Inform** the workers and follow the site's rules; in the EU, video monitoring of workers is regulated by the GDPR and national labour law, and the **AI Act** sets extra obligations for AI used to monitor or evaluate workers.
- Use it to **improve safety** (where and when is equipment often missing?), not to punish individuals.
- **Minimise** what you store: keep counts and statistics, blur faces, delete footage you do not need.
- Keep a **human in the loop** for every decision.

## How to use

```python
# pip install ultralytics huggingface_hub
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

path = hf_hub_download(repo_id="iaac-maai/day4-example-ppe-detector", filename="model.pt")
model = YOLO(path)

result = model.predict("site_photo.jpg", conf=0.25)[0]
for box, cls, conf in zip(result.boxes.xyxy, result.boxes.cls, result.boxes.conf):
    print(model.names[int(cls)], f"{float(conf):.2f}", [round(v) for v in box.tolist()])

result.save("site_photo_detections.jpg")      # the image with the boxes drawn
```

On video, `model.track("video.mp4", conf=0.25, tracker="bytetrack.yaml")` adds an ID to every object across frames.

⚠️ `.pt` files are Python pickles and can execute code when loaded: only load models from sources you trust.

## Reproducibility

The full pipeline (dataset download, training, evaluation, video and SAM examples) is in the GitHub repository: https://github.com/iaac-maai/ai-methods-1-2026. Run `Day_4/docs/03_object_detection_yolo_sam.ipynb` with the packages in the repository's `requirements.txt` and `TRAINING_TIER = "free_cloud"`.

## Authors and licence

AI Methods I course team. Worked example for **AI Methods I: Machine Learning for the Built Environment**, IAAC MaAI 2026/27.

- **Model:** released under **AGPL-3.0**, the licence of the Ultralytics YOLO models it is fine-tuned from. If you build a product or service on it, AGPL-3.0 requires you to share your source code, or you need an Ultralytics Enterprise licence.
- **Training data:** Construction Site Safety Image Dataset, Roboflow Universe, CC BY 4.0.
- **Video frames in this card** (`video_frames.jpg`): adapted from a video by Acabashi, CC BY-SA 4.0 (see *On video*); this figure is shared under the same licence.
