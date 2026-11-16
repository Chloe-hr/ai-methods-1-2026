# Hackathon datasets: 24 options for your autoencoder

This page lists 24 open datasets from construction and the built environment. Each one is big enough to train an autoencoder (AE), can be downloaded today without asking for access, and is unrelated to the datasets used on Days 1 to 4. Pick one for your team, or bring your own dataset if it passes the same checks.

Every link, size and licence below was checked on 16 November 2026, on each dataset's own page or API.

---

## Contents

- [How to pick](#how-to-pick)
- [Already used in this course](#already-used-in-this-course)
- [Overview](#overview)
- [Images: materials and fabrication](#images-materials-and-fabrication) (1 to 5)
- [Drawings and design](#drawings-and-design) (6 to 9)
- [3D and BIM](#3d-and-bim) (10)
- [Tables](#tables) (11 to 14)
- [Sensors and time series](#sensors-and-time-series) (15 to 19)
- [Audio](#audio) (20 and 21)
- [Video](#video) (22 to 24)
- [Alternatives](#alternatives)
- [Left out](#left-out)

---

## How to pick

1. **Start from a subset.** A few thousand samples are enough for a first autoencoder. Download one part of a dataset, not the whole archive; each entry says which part to take first.
2. **For anomaly detection, train on normal samples only.** The AE learns to rebuild "normal". Anything it rebuilds badly (high reconstruction error) is the anomaly. Keep some abnormal samples aside to test it.
3. **Keep the model small.** Your model runs on a free CPU Space. Small inputs (64 to 128 px images, short sensor windows, a few seconds of audio) give a model of a few MB that answers quickly.

"Big enough" here means at least about 2,000 images, 10,000 table rows or 1,000 audio clips, or sensor recordings long enough to cut into thousands of windows. All 24 datasets meet this.

> 🤔 **Think about it:** your app sends the model whatever a user can upload or type into a form: a photo, a CSV of one day, a 10-second clip, a few numbers. Choose the dataset with that input in mind.

> 💡 **Tip:** download data into a `data/` folder that git ignores, and keep it out of iCloud, OneDrive or Dropbox folders. On Day 4 a sync service deleted one team's training images in the middle of a run.

---

## Already used in this course

Do not build your hackathon project on these again.

| Dataset | Used by |
|---|---|
| [Concrete Crack Images for Classification](https://data.mendeley.com/datasets/5y9wdsg2zt/2) (METU; Kaggle [`arunrk7/surface-crack-detection`](https://www.kaggle.com/datasets/arunrk7/surface-crack-detection)) | Day 4, Team 7 |
| [Reinforced concrete structure damage segmentation 2119](https://data.mendeley.com/datasets/2vkm6k4cfg/1) (Mendeley) | Day 4, Team 4 |
| [Roboflow `detr_crack_dataset`](https://universe.roboflow.com/objectdetection-qxiqx/detr_crack_dataset) (Hugging Face [`senthilsk/crack_detection_dataset`](https://huggingface.co/datasets/senthilsk/crack_detection_dataset)) | Day 4, Team 1 |
| [Construction Site Safety](https://www.kaggle.com/datasets/snehilsanyal/construction-site-safety-image-dataset-roboflow) (Roboflow, via Kaggle) | Day 4 lecture, notebook 03 |
| [Church images by architectural style](https://zenodo.org/records/5166987) (Zenodo) | Day 4, Team 5 |
| [Seattle building energy benchmarking](https://www.kaggle.com/datasets/city-of-seattle/sea-building-energy-benchmarking) | Days 1 to 4, Team 6 |
| [UK NEED domestic energy data](https://www.gov.uk/government/collections/national-energy-efficiency-data-framework-need) | Days 1 to 3, Team 4 |
| [NYC DOB facade filings](https://data.cityofnewyork.us/d/xubg-57si) and [PLUTO](https://data.cityofnewyork.us/d/64uk-42ks) | Days 2 and 3, Team 1 |
| House prices and rentals: [King County](https://www.kaggle.com/datasets/harlfoxem/housesalesprediction), [Melbourne](https://www.kaggle.com/datasets/anthonypino/melbourne-housing-market), [Mumbai](https://www.kaggle.com/datasets/harshalhonde/property-rental-listings-dataset), [India](https://www.kaggle.com/datasets/juhibhojani/house-price), housing affordability | Days 1 to 4, several teams; Day 1 lecture |
| [UCI energy efficiency](https://www.kaggle.com/datasets/elikplim/eergy-efficiency-dataset); [UCI household power consumption](https://www.kaggle.com/datasets/uciml/electric-power-consumption-data-set) (daily load profiles) | Day 2 and Day 3 lectures |

That is why this list has no concrete-crack or hard-hat dataset. Three entries are close to earlier work but use new data (#4, #13 and #15). Their entries say what is different.

---

## Overview

| # | Dataset | Type | Size | AE use | Licence |
|---|---|---|---|---|---|
| 1 | [CODD, construction and demolition waste](https://data.mendeley.com/datasets/wds85kt64j/3) | Photos | 3,129 photos, 16,545 objects | Similarity, anomaly | CC BY 4.0 |
| 2 | [MINC-2500, materials in context](https://huggingface.co/datasets/mcimpoi/minc-2500_split_1) | Image patches | 57,500 patches | Similarity, compression | CC BY 4.0 |
| 3 | [RIAWELC weld radiographs](https://github.com/stefyste/RIAWELC) | X-ray images | 24,407 images | Anomaly | Free with citation |
| 4 | [3DPE-Crack22, earthen 3D printing](https://huggingface.co/datasets/ICoM-RWTH/3DPE-Crack22) | Photos (+ video) | 7,593 images | Anomaly | CC BY 4.0 |
| 5 | [I3DCP, 3D concrete printing](https://zenodo.org/records/20628313) | Texture images | 537 real + 6,000 synthetic | Anomaly | CC BY-NC-SA 4.0 |
| 6 | [CubiCasa5K floor plans](https://zenodo.org/records/2613548) | Plan images | 5,000 plans | Similarity, generative, denoising | CC BY-NC-SA 4.0 |
| 7 | [ResPlan floor plans](https://github.com/m-agour/ResPlan) | Vector plans + graphs | 17,000 plans | Generative, similarity | CC BY 4.0 |
| 8 | [BridgeNet](https://huggingface.co/datasets/lazlo-bleker/bridge-net) | Graphs, renders, meshes | 20,000 bridges | Generative | MIT |
| 9 | [TopoDiff topologies](https://decode.mit.edu/projects/topodiff/) | 64 × 64 grids | 33,000 samples | Generative | Not stated |
| 10 | [IFCNet BIM objects](https://ifcnet.e3d.rwth-aachen.de/) | IFC, meshes, point clouds, renders | 19,613 objects | Similarity | Not stated |
| 11 | [Swiss Dwellings](https://zenodo.org/records/7788422) | Table + geometry | 45,176 apartments | Similarity, anomaly | CC BY 4.0 |
| 12 | [ASHRAE Thermal Comfort Database II](https://datadryad.org/dataset/doi:10.6078/D1F671) | Table | 109,033 rows | Anomaly, compression | CC0 |
| 13 | [NYC DOB Permit Issuance](https://data.cityofnewyork.us/d/ipu4-2q9a) | Table | 3,991,389 rows | Anomaly | NYC Open Data terms |
| 14 | [OSHA Severe Injury Reports](https://www.osha.gov/severe-injury-reports) | Table + text | about 100,000 reports | Similarity | Public domain |
| 15 | [Building Data Genome 2](https://github.com/buds-lab/building-data-genome-project-2) | Hourly meters | 3,053 meters | Anomaly | CC BY 4.0 |
| 16 | [LBNL Building 59](https://datadryad.org/dataset/doi:10.7941/D1N33Q) | 300+ building sensors | 3 years | Anomaly | CC0 |
| 17 | [LBNL HVAC fault datasets](https://faultdetection.lbl.gov/data/) | Labelled HVAC series | 8 systems | Anomaly | CC BY 4.0 |
| 18 | [KW51 railway bridge](https://zenodo.org/records/3745914) | Bridge monitoring | 15 months | Anomaly | CC BY-NC-SA 4.0 |
| 19 | [VTT-ConIoT](https://zenodo.org/records/4683703) | Worker motion sensors | 13 workers, 16 activities | Similarity, anomaly | CC BY 4.0 |
| 20 | [SONYC-UST v2](https://zenodo.org/records/3966543) | Street audio | 18,510 clips | Anomaly, denoising | CC BY 4.0 |
| 21 | [DCASE 2024 machine sounds](https://zenodo.org/records/11259435) | Machine audio | 1,000 clips per machine | Anomaly | CC BY-NC-SA 4.0 |
| 22 | [Earthmoving videos](https://data.mendeley.com/datasets/fyw6ps2d2j/1) | Video | 10 videos | Anomaly, similarity | CC BY 4.0 |
| 23 | [IITU construction time-lapse](https://data.mendeley.com/datasets/ww7rt2rnk6/1) | Video | 12 months | Compression, similarity | CC BY 4.0 |
| 24 | [VolvoConstAct hand signals](https://mmahdavian.github.io/ls_har/) | Video + skeletons | 17 signals | Similarity, anomaly | Not stated |

**About the licences.** All of them allow coursework.
- **NC (non-commercial):** don't use the data in a commercial product.
- **SA (share-alike):** anything you publish from the data, such as a derived dataset on Hugging Face, must use the same licence.
- **"Not stated":** cite the paper and don't re-upload the raw data anywhere.

---

## Images: materials and fabrication

### 1. CODD, construction and demolition waste

Photos from a recycling plant in Cyprus, with every piece of waste outlined and labelled as one of ten materials: brick, concrete, tile, wood, pipe, plastic, general waste, foam insulation, stone or plasterboard.

- **Size:** 3,129 photos (1920 × 1200) with 16,545 labelled objects. Each object has a box and a polygon (PASCAL VOC XML). Train, validation and test splits are included.
- **Get it:** [Mendeley Data, DOI 10.17632/wds85kt64j.3](https://data.mendeley.com/datasets/wds85kt64j/3), then *Download All*.
- **Licence:** CC BY 4.0.
- **Cite:** CODD, Mendeley Data, DOI 10.17632/wds85kt64j.3.
- **AE task:** crop every labelled object (about 16,500 crops), resize to 64 to 128 px and train a convolutional AE.
  - Similarity search: use the latent vectors to find the most similar pieces.
  - Anomaly detection: train on one material, then flag foreign pieces by reconstruction error.
- **App idea:** upload a photo of a piece of rubble. The app returns the five most similar pieces and their material, for sorting reclaimed material on a demolition site.

### 2. MINC-2500, materials in context

Close-up patches of real surfaces cut from photos, in 23 material classes. 13 classes are building materials: brick, carpet, ceramic, glass, metal, mirror, painted, plastic, polished stone, stone, tile, wallpaper and wood.

- **Size:** 57,500 patches of 362 × 362 px (48,875 train, 2,875 validation, 5,750 test), 2,500 per class. Download 2.35 GB (parquet).
- **Get it:** Hugging Face [`mcimpoi/minc-2500_split_1`](https://huggingface.co/datasets/mcimpoi/minc-2500_split_1). Run `pip install datasets`, then `load_dataset("mcimpoi/minc-2500_split_1")`. Alternatively, download the parquet files with `snapshot_download(..., repo_type="dataset")` and read them with `polars`, which is already in the course environment.
- **Licence:** CC BY 4.0 (stated on the mirror).
- **Cite:** Bell, Upchurch, Snavely and Bala (2015), "Material Recognition in the Wild with the Materials in Context Database", CVPR.
- **AE task:** train a convolutional AE on the 13 building-material classes (32,500 patches) at 64 to 128 px. Use it for similarity search in latent space, or to compress a material library.
- **App idea:** upload a photo of a facade or a floor. The app returns the closest material patches and the material name ("find a tile that looks like this").

> 💡 **Tip:** drop the classes that are not building materials (food, foliage, hair, skin, sky, water, fabric, leather, paper, other) before training.

### 3. RIAWELC weld radiographs

X-ray images of steel weld seams, cut into 224 × 224 px patches. There are four classes: no defect, porosity, crack and lack of penetration.

- **Size:** 24,407 greyscale PNG images. The repository is about 470 MB.
- **Get it:** [github.com/stefyste/RIAWELC](https://github.com/stefyste/RIAWELC), with `git clone https://github.com/stefyste/RIAWELC`. The images are in `Dataset_partitioned/`.
- **Licence:** no formal licence. The README says the data is released freely and asks you to cite two papers.
- **Cite:** the two papers listed in the [README](https://github.com/stefyste/RIAWELC), starting with Totino, Spagnolo and Perri (2023), "RIAWELC: a novel dataset of radiographic images for automatic weld defects classification".
- **AE task:** train on "no defect" only. Reconstruction error then flags pores, cracks and lack of penetration.
- **App idea:** upload a weld radiograph. The app returns an anomaly score and a heatmap of where the AE fails, as a quality check for structural steel.

> ⚠️ **Watch out:** no licence is stated. Cite the papers and don't re-upload the images.

### 4. 3DPE-Crack22, earthen 3D printing on site

On-site photos of large earthen walls printed by RWTH Aachen, with cracks labelled in YOLO format (classes `crack_b`, `crack_s` and `crack_shadow`). The repository also holds GoPro footage of the printing.

- **Size:** 7,593 images of 640 × 640 px, made from about 2,000 original photos; the training split includes brightness-shifted copies. The core download is about 380 MB. Optional: frames at 1 per second (about 2 GB) and 1-minute video clips (about 28 GB) in `org_footages/`.
- **Get it:** [Hugging Face `ICoM-RWTH/3DPE-Crack22`](https://huggingface.co/datasets/ICoM-RWTH/3DPE-Crack22), then:
  ```python
  from huggingface_hub import snapshot_download
  snapshot_download("ICoM-RWTH/3DPE-Crack22", repo_type="dataset",
                    allow_patterns=["train/*", "valid/*", "test/*", "data.yaml"])
  ```
- **Licence:** CC BY 4.0.
- **Cite:** the paper linked on the [dataset page](https://huggingface.co/datasets/ICoM-RWTH/3DPE-Crack22) (RWTH Aachen, ICoM, 2025).
- **AE task:** use the YOLO boxes to cut crack-free patches as "normal" training data. Cracks and print defects then stand out by reconstruction error. Alternatively, train on the 1-per-second frames for a print-monitoring AE.
- **App idea:** upload a photo of a printed wall. The app returns a heatmap of suspicious zones.

> ⚠️ **Watch out:** three Day 4 teams detected cracks. This dataset is new (printed earth, on site), so frame your project as print monitoring, not as crack detection again. Skip `org_footages/videos_*` (28 GB) unless you need video.

### 5. I3DCP, 3D concrete printing texture windows

Greyscale texture windows of freshly printed concrete layers, from cameras on a 3D concrete printer in Nantes. The classes describe the extrusion: good, fluid, dry, tearing and (in sub-dataset 2) crushed.

- **Size:** 537 real windows (111 + 426, 82 MB) plus 6,000 synthetic windows generated with StyleGAN3 (1,200 per class, 2.35 GB).
- **Get it:** [Zenodo 20628313](https://zenodo.org/records/20628313). Take `subdataset1.zip`, `subdataset2.zip` and `synthetic images.zip`, and skip the two model-weight zips (17 GB each).
- **Licence:** CC BY-NC-SA 4.0.
- **Cite:** Rill-García et al. (2022), "Inline monitoring of 3D concrete printing using computer vision", Additive Manufacturing.
- **AE task:** train on "good" windows (synthetic and real). Reconstruction error then flags fluid, dry or torn layers.
- **App idea:** upload a photo of a printed layer. The app says "good" or "suspicious" and shows where.

> ⚠️ **Watch out:** most of the data is synthetic. Report your test score on the real windows only.

---

## Drawings and design

### 6. CubiCasa5K floor plans

5,000 floor plan images of mostly Finnish homes, with SVG annotations of more than 80 element types (walls, rooms, doors, windows, fixtures).

- **Size:** 5,000 plans (4,200 train, 400 validation, 400 test), in one 5.47 GB zip.
- **Get it:** [Zenodo 2613548](https://zenodo.org/records/2613548), direct link `https://zenodo.org/records/2613548/files/cubicasa5k.zip?download=1`.
- **Licence:** CC BY-NC-SA 4.0.
- **Cite:** Kalervo et al. (2019), "CubiCasa5K: A Dataset and an Improved Multi-Task Model for Floorplan Image Analysis", SCIA.
- **AE task:** resize the plans to 128 × 128 px greyscale and train a convolutional AE or VAE. Uses:
  - similarity search between plans;
  - interpolation between two plans;
  - denoising scanned plans (add noise, then train the AE to remove it).
- **App idea:** upload a floor plan. The app returns the five most similar plans, or a slider that morphs one plan into another.

> ⚠️ **Watch out:** 5.5 GB in one file. Download it on campus Wi-Fi and unzip it once.

### 7. ResPlan floor plans

17,000 residential floor plans as clean vector geometry in metres: walls, doors, windows, balconies and rooms with 17 room types. Each plan comes with a graph of how its rooms connect.

- **Size:** 17,000 plans (13,053 train, 1,632 validation, 1,632 test) plus 683 augmented copies. The repository is about 285 MB.
- **Get it:** [Kaggle `resplan/resplan`](https://www.kaggle.com/datasets/resplan/resplan), with `kagglehub.dataset_download("resplan/resplan")`, which includes the graphs. Alternatively, use GitHub [`m-agour/ResPlan`](https://github.com/m-agour/ResPlan), whose `resplan_utils.py` has a `plot_plan` function.
- **Licence:** CC BY 4.0 for the data, MIT for the code.
- **Cite:** Abouagour and Garyfallidis (2025), "ResPlan: A Large-Scale Vector-Graph Dataset of 17,000 Residential Floor Plans", arXiv 2508.14006.
- **AE task:** rasterise each plan to a 64 to 128 px image (for example, one colour per room type) and train a VAE. Use it to generate new layouts, interpolate between plans or find similar plans.
- **App idea:** a form with apartment size and number of rooms returns a generated plan. Or upload a sketch and get the closest real plans.

> 💡 **Tip:** all plans come from South Asian housing markets, so the layouts follow local conventions.

### 8. BridgeNet

20,000 bridges form-found with Combinatorial Equilibrium Modelling (TUM). Each bridge has:
- a pin-jointed wireframe (nodes and edges, with forces, stresses and deflection);
- a 3D mesh (OBJ);
- two renders (JPG);
- its input parameters (span, number of cables and bays, mid-span height, deck width, twist and others).

- **Size:** 20,000 bridges. The full repository is 74 GB because each mesh is about 5.6 MB. The renders are about 1.9 GB and the parameters 8 MB.
- **Get it:** [Hugging Face `lazlo-bleker/bridge-net`](https://huggingface.co/datasets/lazlo-bleker/bridge-net). Download only renders and parameters, here for the first 10,000 bridges (folders starting with `0`):
  ```python
  from huggingface_hub import snapshot_download
  snapshot_download("lazlo-bleker/bridge-net", repo_type="dataset",
                    allow_patterns=["0*_images.jpg", "0*_input_params.json"])
  ```
- **Licence:** MIT.
- **Cite:** Bleker, Güneş and D'Acunto (2025), "BridgeNet: A Dataset of Graph-based Bridge Structural Models for Machine Learning Applications", arXiv 2512.14496.
- **AE task:** train a VAE on the renders (64 to 128 px) or on the parameter and force vectors. This gives a latent design space of bridges that sliders can move through.
- **App idea:** sliders (latent dimensions, or span and height) return a generated bridge, plus the closest real bridge and its parameters.

> ⚠️ **Watch out:** never download the whole repository (74 GB, 120,000 files). Always use `allow_patterns`.

### 9. TopoDiff structural topologies

2D structures from topology optimisation (SIMP method, ToPy) for random loads and supports. Each sample is a 64 × 64 grid with 6 channels: optimal topology, volume fraction, von Mises stress, strain energy density and the loads.

- **Size:** 33,000 samples in the main set. There are also sets of non-optimal structures (12,000 and 30,000) and two test sets.
- **Get it:** the Dropbox folder linked on the [project page](https://decode.mit.edu/projects/topodiff/). `dataset_1_diff` is the main set.
- **Licence:** not stated.
- **Cite:** Mazé and Ahmed (2023), "Diffusion Models Beat GANs on Topology Optimization", AAAI (arXiv 2208.09591).
- **AE task:** train a convolutional VAE on the topology channel, optionally conditioned on the loads, to generate and interpolate structures. Anomaly detection: flag non-optimal shapes, using the extra sets as test data.
- **App idea:** choose a load position and a volume fraction in a form. The app returns a generated structural topology.

> ⚠️ **Watch out:** no licence is stated. Cite the paper and don't re-upload the data.

---

## 3D and BIM

### 10. IFCNet BIM objects

Single building elements taken from BIM models as IFC files, each labelled with its IFC class (beams, doors, walls, slabs, ducts, pipes, furniture and more). IFCNetCore, a 20-class subset, also comes as meshes, point clouds and multi-view renders.

- **Size:** 19,613 objects in 65 classes (full IFC set, 197 MB). IFCNetCore comes as multi-view PNG (466 MB), PLY point clouds (215 MB) or OBJ meshes (145 MB).
- **Get it:** direct links on [ifcnet.e3d.rwth-aachen.de](https://ifcnet.e3d.rwth-aachen.de/), for example `https://ifcnet.e3d.rwth-aachen.de/IFCNetCorePng.zip`.
- **Licence:** not stated (benchmark by RWTH Aachen; the model code is MIT).
- **Cite:** Emunds et al. (2021), "IFCNet: A Benchmark Dataset for IFC Entity Classification", arXiv 2106.09712.
- **AE task:** train a convolutional AE on the multi-view renders (easiest), or a point-cloud AE on the PLY files (advanced). Uses:
  - similarity search between BIM objects;
  - anomaly detection: flag objects whose geometry doesn't fit their IFC class, which catches mislabelled elements in a BIM model.
- **App idea:** upload a render or screenshot of a component. The app returns the most similar BIM objects and their IFC class.

> ⚠️ **Watch out:** no licence is stated. Cite the paper and don't re-upload the data.

---

## Tables

### 11. Swiss Dwellings

45,176 Swiss apartments in about 3,100 buildings, digitised from real plans by Archilyse. It includes:
- geometry of walls, openings and rooms (WKT);
- simulation results per room: views (sky, greenery, buildings and more), sun and daylight at the solstices and equinox, traffic and train noise, and layout and connectivity metrics.

- **Size:** about 370,000 rooms, in one 932 MB zip (`geometries.csv`, `simulations.csv`, `locations.csv`, `location_ratings.csv`).
- **Get it:** [Zenodo 7788422](https://zenodo.org/records/7788422), direct link `https://zenodo.org/records/7788422/files/swiss-dwellings-v3.0.0.zip?download=1`.
- **Licence:** CC BY 4.0.
- **Cite:** Swiss Dwellings v3.0.0, Zenodo, DOI 10.5281/zenodo.7788422.
- **AE task:** aggregate the simulation columns per apartment and train a tabular AE. Its latent vector is an "apartment fingerprint" for similarity search and outlier detection (apartments with unusual light, noise or layout).
- **App idea:** enter an apartment's numbers in a form, or pick one. The app returns the most similar apartments and how unusual this one is.

### 12. ASHRAE Global Thermal Comfort Database II

Field measurements of indoor climate (air and radiant temperature, humidity, air speed), clothing and activity, each paired with what the occupant reported (thermal sensation, preference, comfort). The measurements come from buildings worldwide.

- **Size:** 109,033 rows × 53 columns, 3 MB compressed.
- **Get it:** [GitHub `CenterForTheBuiltEnvironment/ashrae-db-II`](https://github.com/CenterForTheBuiltEnvironment/ashrae-db-II), loaded directly with:
  ```python
  pd.read_csv("https://raw.githubusercontent.com/CenterForTheBuiltEnvironment/ashrae-db-II/master/v2.1.0/db_measurements_v2.1.0.csv.gz")
  ```
  The data is also on [Dryad](https://datadryad.org/dataset/doi:10.6078/D1F671).
- **Licence:** CC0 (Dryad).
- **Cite:** Földváry Ličina et al. (2018), "Development of the ASHRAE Global Thermal Comfort Database II", Building and Environment 142.
- **AE task:** train a tabular AE on the measured and reported columns. Use the anomaly score to find unusual comfort situations or sensor errors, or compress to 2 or 3 latent dimensions to map comfort types.
- **App idea:** a form with today's room readings and how people feel. The app returns "typical" or "unusual", plus the closest recorded situations.

> ⚠️ **Watch out:** many columns are empty for many rows. Choose columns with good coverage first.

### 13. NYC DOB Permit Issuance

Every construction permit issued in the older system (BIS) of the New York City Department of Buildings. Columns include job type, work type, permit type, building type, borough and dates (filing, issuance, expiration, job start).

- **Size:** 3,991,389 rows on 16 November 2026, updated daily.
- **Get it:** [NYC Open Data, `ipu4-2q9a`](https://data.cityofnewyork.us/d/ipu4-2q9a). A sample through the API:
  ```python
  pd.read_csv("https://data.cityofnewyork.us/resource/ipu4-2q9a.csv?$limit=200000")
  ```
  For the full CSV, export it from the [dataset page](https://data.cityofnewyork.us/d/ipu4-2q9a).
- **Licence:** NYC Open Data terms of use.
- **AE task:** encode the categories and durations (for example, days from filing to issuance) and train a tabular AE. Its anomaly score flags unusual permits.
- **App idea:** a form for a new permit (job type, borough, building type, dates). The app says how unusual it is.

> ⚠️ **Watch out:** the table holds names and phone numbers of permittees and owners. Drop those columns before training, and never store them in your app.

> ⚠️ **Watch out:** Team 1 used another Department of Buildings dataset on Days 2 and 3 (facade filings). This one is permits. Say what is new in your README.

> 💡 **Tip:** most recent permits are issued in the newer DOB NOW system (dataset `rbx6-tga4`). Use it if you need permits after the switch.

### 14. OSHA Severe Injury Reports

Every severe work injury (amputation, hospitalisation, loss of an eye) reported to US federal OSHA since 2015. Each report has the date, the employer's industry (NAICS code), location, nature of the injury, part of body, event, source and a short narrative of what happened.

- **Size:** January 2015 to November 2025, about 100,000 reports. Construction (NAICS codes starting with 23) was about 18 per cent of reports from 2015 to 2021.
- **Get it:** [osha.gov/severe-injury-reports](https://www.osha.gov/severe-injury-reports), file `January2015toNovember2025.zip`. Download it in a browser, because the site blocks scripts.
- **Licence:** US government data, public domain.
- **AE task:** one-hot encode the coded columns, add a TF-IDF of the narrative, and train an AE. The resulting incident embeddings support similarity search and clustering of accident types, and an anomaly score for unusual combinations.
- **App idea:** type a short description of an incident. The app returns the most similar past construction accidents and their outcomes.

> ⚠️ **Watch out:** filter on NAICS 23 for construction. The data only covers states under federal OSHA.

---

## Sensors and time series

### 15. Building Data Genome 2

Hourly readings from 3,053 energy meters in 1,636 non-residential buildings, 2016 to 2017. Meters cover electricity, chilled water, steam, hot water, gas, water, irrigation and solar. Weather and building metadata are included.

- **Size:** 595 MB (Zenodo zip).
- **Get it:** [Kaggle `claytonmiller/buildingdatagenomeproject2`](https://www.kaggle.com/datasets/claytonmiller/buildingdatagenomeproject2), with `kagglehub.dataset_download("claytonmiller/buildingdatagenomeproject2")`, [Zenodo 3887306](https://zenodo.org/records/3887306), or GitHub [`buds-lab/building-data-genome-project-2`](https://github.com/buds-lab/building-data-genome-project-2).
- **Licence:** CC BY 4.0.
- **Cite:** Miller et al. (2020), "The Building Data Genome Project 2, energy meter data from the ASHRAE Great Energy Predictor III competition", Scientific Data 7.
- **AE task:** cut each meter into 24-hour profiles (more than 2 million days) and train an AE. Reconstruction error flags abnormal days (faults, holidays, meter errors), and the latent space groups buildings by how they use energy.
- **App idea:** upload one day of hourly readings as a CSV. The app returns "normal" or "abnormal" and draws the expected profile on top.

> ⚠️ **Watch out:** the Day 3 lecture clustered the daily load profiles of one household. This dataset covers 1,636 non-residential buildings and a new task (anomaly detection with an AE). Say what is new in your README.

### 16. LBNL Building 59, three years of operation

Three years (2018 to 2020) of data from an office building in Berkeley, California: more than 300 sensors and meters on two office floors. It covers whole-building and end-use energy, HVAC operation, indoor and outdoor conditions, and occupant counts, including the 2020 COVID-19 shutdown.

- **Size:** 263 MB zip (2.4 GB of CSV once unzipped).
- **Get it:** [Dryad, doi:10.7941/D1N33Q](https://datadryad.org/dataset/doi:10.7941/D1N33Q), file `Building_59.zip`. Download it in a browser.
- **Licence:** CC0.
- **Cite:** Luo et al. (2022), "A three-year dataset supporting research on building energy management and occupancy analytics", Scientific Data 9, 156.
- **AE task:** train a multivariate AE on daily or hourly windows of many sensors. The reconstruction error per sensor shows where the building misbehaves, and the March 2020 shutdown is a natural test anomaly.
- **App idea:** upload one day of sensor data. The app returns an anomaly score and the sensors that look wrong.

### 17. LBNL HVAC fault detection datasets

HVAC operation data with known faults inserted at several severity levels (for example, stuck dampers or valves and biased sensors), plus fault-free periods. There are eight sets:
- single-duct air handling unit (AHU);
- dual-duct AHU;
- rooftop units (simulation, lab and field data);
- fan-powered VAV terminal units;
- fan coil units;
- chiller plant;
- boiler plant;
- a combined AHU and rooftop-unit set.

- **Size:** differs per set (CSV files); each set is a separate download.
- **Get it:** [faultdetection.lbl.gov/data](https://faultdetection.lbl.gov/data/), then choose a system.
- **Licence:** CC BY 4.0 (DOI 10.25984/1881324).
- **AE task:** train on fault-free rows only, so that reconstruction error detects the faults. The labels let you measure precision and recall honestly.
- **App idea:** upload a CSV of AHU readings. The app returns a fault alarm per time step and the signals that drifted.

> 💡 **Tip:** start with the single-duct AHU set.

### 18. KW51 railway bridge

15 months of monitoring of a steel railway bridge in Leuven (KU Leuven), October 2018 to January 2020. It covers accelerations, strains, displacements, temperature and humidity, ambient vibrations and train passages, plus the natural frequencies identified every hour. The bridge was strengthened between May and September 2019.

- **Size:** the tracked natural frequencies are 13 MB. The raw data comes in monthly zips of 0.2 to 4.3 GB (45.6 GB in total).
- **Get it:** [Zenodo 3745914](https://zenodo.org/records/3745914). Take `trackedmodes.zip` first: `https://zenodo.org/records/3745914/files/trackedmodes.zip?download=1`.
- **Licence:** CC BY-NC-SA 4.0.
- **Cite:** Maes and Lombaert (2021), "Monitoring Railway Bridge KW51 Before, During, and After Retrofitting", Journal of Bridge Engineering 26(3).
- **AE task:** train on the natural frequencies (with temperature) from before the retrofit. The AE learns the normal seasonal variation, and after the retrofit the reconstruction error jumps. This is the textbook set-up for structural health monitoring.
- **App idea:** enter today's measured frequencies and temperature. The app answers "same structure" or "changed structure" and shows the error trend.

> ⚠️ **Watch out:** only download the raw monthly zips if you need the raw signals.

### 19. VTT-ConIoT, construction worker motion

13 people performing 16 construction-work activities while wearing three motion sensors (IMU), recorded by VTT Finland and checked against a pilot on a real construction site. It also includes body keypoints from video.

- **Size:** 363 MB zip.
- **Get it:** [Zenodo 4683703](https://zenodo.org/records/4683703), direct link `https://zenodo.org/records/4683703/files/VTT_ConIot_Dataset.zip?download=1`.
- **Licence:** CC BY 4.0.
- **Cite:** "Introducing VTT-ConIot: A Realistic Dataset for Activity Recognition of Construction Workers Using IMU Devices", Sustainability 14(1), 220 (2022).
- **AE task:** cut the signals into windows of 2 to 5 seconds and train a 1D convolutional AE. Its latent space groups activities without labels, and unusual or risky movements get a high reconstruction error.
- **App idea:** upload a short sensor recording as a CSV. The app returns the activity it most resembles and how unusual it is.

---

## Audio

### 20. SONYC-UST v2, New York street sound

10-second recordings from acoustic sensors on New York City streets. Volunteers tagged them with 23 sound classes chosen with the city's environmental agency. Construction-noise classes include rock drill, jackhammer, hoe-ram, pile driver, chainsaw, small/medium and large rotating saws, and engines of three sizes.

- **Size:** 18,510 clips (51 hours) in 19 archives of about 720 MB each (13.3 GB in total), plus `annotations.csv` (14.5 MB).
- **Get it:** [Zenodo 3966543](https://zenodo.org/records/3966543): `annotations.csv` and two or three `audio-*.tar.gz` parts.
- **Licence:** CC BY 4.0.
- **Cite:** Cartwright et al. (2020), "SONYC-UST-V2: An Urban Sound Tagging Dataset with Spatiotemporal Context", DCASE Workshop.
- **AE task:** train a convolutional AE on log-mel spectrograms. Uses:
  - anomaly detection: machinery against normal street sound;
  - denoising: remove traffic from a construction recording;
  - similarity search between noise types.
- **App idea:** upload a 10-second clip from a site. The app answers "construction noise detected, sounds like a jackhammer" and plots loudness over time, for noise compliance on site.

> ⚠️ **Watch out:** download two or three parts first (about 2 GB, roughly 3,000 clips). Use `annotations.csv` to find the construction-noise clips.

### 21. DCASE 2024 machine sounds

Machine sounds recorded for anomaly detection challenges. There are two groups:
- **Fabrication and site machines:** 3D printer, robotic arm, air compressor, hovering drone; also brushless motor and scanner.
- **Building-services machines:** fan and valve; also bearing, gearbox and slider.

- **Size:** 1,000 normal training clips of 6 to 10 seconds per machine, 180 to 380 MB per zip.
- **Get it:**
  - [Zenodo 11259435](https://zenodo.org/records/11259435), for example `eval_data_RoboticArm_train_r2.zip` or `eval_data_3DPrinter_train_r2.zip`;
  - [Zenodo 10902294](https://zenodo.org/records/10902294): `dev_fan.zip`, `dev_valve.zip`.
- **Licence:** CC BY-NC-SA 4.0.
- **Cite:** Nishida et al. (2024), "Description and Discussion on DCASE 2024 Challenge Task 2: First-Shot Unsupervised Anomalous Sound Detection for Machine Condition Monitoring", arXiv 2406.07250.
- **AE task:** the official challenge baseline is exactly this: an AE on log-mel frames, trained on normal sounds, with reconstruction error as the anomaly score.
- **App idea:** upload a recording of a robotic arm. The app answers "healthy" or "check this machine" and gives the score.

> 💡 **Tip:** the development zips (fan, valve and the others) contain test clips whose file names say "normal" or "anomaly", so you can measure your detector. The 3D printer and robotic arm zips only contain normal clips.

---

## Video

### 22. Earthmoving videos

10 videos filmed at ground level of excavators and dump trucks working together, with per-frame boxes, object identities and activity labels (University of Illinois).

- **Size:** 2.0 GB (`Data.zip`).
- **Get it:** [Mendeley Data, DOI 10.17632/fyw6ps2d2j.1](https://data.mendeley.com/datasets/fyw6ps2d2j/1).
- **Licence:** CC BY 4.0.
- **Cite:** Roberts and Golparvar-Fard (2019), "End-to-end vision-based detection, tracking and activity analysis of earthmoving equipment filmed at ground level", Automation in Construction 105.
- **AE task:** crop the excavator box in each frame, stack short clips (for example, 8 frames at 64 px) and train a convolutional AE. Latent trajectories separate work cycles from idle time, and unusual motion gets a high reconstruction error.
- **App idea:** upload a short site clip. The app returns a working/idle timeline and the number of work cycles, as a productivity measure.

### 23. IITU construction time-lapse

Video from a fixed street-level camera filming one reinforced-concrete building in Almaty go up over 12 months (May 2025 to May 2026), in all weather, day and night. Version 1 has the raw video only; labelled frames (floor count, construction stage) are announced for a later version.

- **Size:** 5.1 GB zip.
- **Get it:** [Mendeley Data, `ww7rt2rnk6`](https://data.mendeley.com/datasets/ww7rt2rnk6/1).
- **Licence:** CC BY 4.0.
- **Cite:** IITU Multi-Temporal Building Floor-Count Dataset, Mendeley Data, DOI 10.17632/ww7rt2rnk6.1.
- **AE task:** sample frames (for example, one per hour), train a convolutional AE and plot the latent vectors over time. Construction progress becomes a path, weather and night form their own clusters, and unexpected events on site get a high reconstruction error.
- **App idea:** upload a photo of the site. The app estimates the construction stage and date from the nearest frames in latent space.

> 💡 **Tip:** extract frames once with OpenCV (Day 4, notebook 02) and work with the frames from then on.

### 24. VolvoConstAct, construction site hand signals

People performing 17 hand signals and machine commands used on construction sites, recorded as RGB video, skeleton keypoints and depth. The signals are: All OK?, Arm Circle, Back Up, Come, Do You See Me?, Emergency, Excavator Swing Left and Right, Follow Me, Load Lift Up and Down, Slow Down, Stop, Turn Left and Right, and Wheel Loader Tilt Up and Down.

- **Size:** 9.7 GB (`volvo.zip`).
- **Get it:** the SFU download link on the [project page](https://mmahdavian.github.io/ls_har/).
- **Licence:** not stated.
- **Cite:** Mahdavian et al. (2024), "LS-HAR: Language Supervised Human Action Recognition with Salient Fusion, Construction Sites as a Use-Case", arXiv 2410.01962.
- **AE task:** train a sequence AE on the skeleton keypoints; this is light, with no images needed. It gives a latent space of gestures, and unknown or sloppy signals get a high reconstruction error.
- **App idea:** upload a short clip or a keypoint file. The app answers "this looks like Stop" and says how confident it is that the signal is valid.

> ⚠️ **Watch out:** 9.7 GB in one zip, so start the download early. No licence is stated: cite the paper and don't re-upload the data.

---

## Alternatives

These also passed the checks but are not counted in the 24.

- **FloorPlanCAD:** more than 15,000 annotated CAD floor plans (SVG and PNG), residential to commercial, CC BY-NC 4.0, [floorplancad.github.io](https://floorplancad.github.io/). Use it if your team wants CAD drawings, for example for symbol similarity or drawing denoising.
- **TOP topology dataset:** 10,000 topology-optimisation runs, 100 iterations each, on a 40 × 40 grid. About 3 GB on Yandex Disk, no licence stated, [github.com/ISosnovik/top](https://github.com/ISosnovik/top).
- **UR3 CobotOps:** 7,409 rows × 20 features from a UR3 collaborative robot (joint currents, temperatures, speeds, protective stops, grip losses). CC BY 4.0, [UCI dataset 963](https://archive.ics.uci.edu/dataset/963/ur3+cobotops). Small, but directly about robotic fabrication.
- **Guangzhou GPR scans:** 3.8 GB of raw ground-penetrating radar scans (tunnel linings, buried pipes, rebar in slabs). CC BY 4.0, [Zenodo 14637589](https://zenodo.org/records/14637589). Advanced: the files are raw IDS `.dt`, so you need a reader before you can make images.

---

## Left out

- **Close to earlier work in this course:**
  - [SDNET2018](https://digitalcommons.usu.edu/all_datasets/48) (concrete cracks again);
  - [dacl10k bridge damage](https://huggingface.co/datasets/Voxel51/dacl10k) (same damage classes as Day 4 Team 4);
  - [Hard Hat Workers](https://huggingface.co/datasets/keremberke/hard-hat-detection) (same task as the Day 4 lecture).
- **Access must be requested and approved first:** MOCS (moving objects on construction sites), ACID (construction machines), [PEER Hub ImageNet](https://apps.peer.berkeley.edu/phi-net/) (structural damage) and Sewer-ML (sewer inspection).
- **Too heavy for a one-week hackathon:**
  - [wood surface defects](https://zenodo.org/records/4694695) (157 GB);
  - [TBBR thermal bridges on rooftops](https://zenodo.org/records/7022736) (68.5 GB).
- **Only a raw database export:** [RoofN3D](https://roofn3d.gis.tu-berlin.de/) roof point clouds.
