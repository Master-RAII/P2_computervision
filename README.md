# P2 — Machine Learning and Deep Learning Course Notes

The notebooks for P2 of the MSc Responsible AI Innovation: neural networks, training and evaluation, and
computer vision. P2 continues where P1 (univariate analysis through `MLPClassifier`) stops — notebook `01`
opens by explaining what happened inside the `.fit()` call at the end of P1's last notebook.

| Week | Lesson 1 | Lesson 2 |
|---|---|---|
| 1 | `01` Neural Network Foundations — inside `.fit()`; `00` is homework before it | `02` PyTorch Foundations — tensors, datasets, the training loop |
| 2 | `03` Convolutional Neural Networks — convolution, pooling, a first CNN | `04` Auditing an Image Dataset — chapter 1 of the image assignment; `sohas_classification.zip` downloaded before the lesson, sections 12 and 13 at home |
| 3 | `05` Evaluation Metrics — precision, recall, thresholds; `06` Class Imbalance as self-study | `07` Training & Learning Curves — learning rate, epochs, curves |
| 4 | `08` Regularization — weight decay, dropout | `09` Data Augmentation — more from the same images |
| 5 | `10` Hyperparameter Tuning — splits, search, cross-validation | `11` Transfer Learning — pretrained ResNet50; **part A: classification** |
| 6 | `12` Annotations & Metrics — boxes, IoU | `13` Object Detection — the detector interface, NMS, thresholds; **part B: YOLO** |

Lessons are about 2 hours each, two per week. Weeks 8 and 9 have no lessons: they are for the two final assignments, both handed in at the end of week 9.



## Data

### Overview

Two kinds of dataset: the ones the theory notebooks download for themselves, and the ones your teacher
hands out for the final assignments. The rows in **bold** are the ones a grade rests on.

| Dataset | Used in | Role | Size | Where it goes |
|---|---|---|---|---|
| MNIST | `02`, `03`, `07`, `08`, `09`, `11` | theory and practice; downloaded by the notebook | about 12 MB | `data/raw/` |
| CIFAR-10 | `03`, `04` | theory and practice; downloaded by the notebook | about 170 MB | `data/raw/` |
| The messy image folder, built from CIFAR-10 | `04` | practising the audit | a few MB | `data/messy_images/` |
| **Sohas classification** | `04`, `11` | **image assignment, part A**; `sohas_classification.zip` | about 150 MB | `data/raw/sohas_classification/` |
| **Sohas detection** | `13` | **image assignment, part B**; `sohas_detection.zip` | about 1.8 GB | `data/raw/sohas_detection/` |

### Getting the datasets from your teacher

The four zips named in the table are not in this repository and are not built by it: your teacher
prepares them and shares them through the course platform. For each one:

1. download the zip and put it, unchanged, in `data/raw/`;
2. unpack it there, so that the zip and its folder sit side by side (`data/raw/scfd.zip` and
   `data/raw/scfd/`) — notebook `04`, section 12, shows how for the first one;
3. never edit, move or delete files inside the unpacked folder. Your decisions about the data live in your
   code and your report.

Each zip holds a `LICENSE.md` with its source, licence and citation; keep it with the data. Everything
under `data/` is in `.gitignore`, so no dataset ends up in git, and the zips may not be passed on outside
the course. The tabular crime-safety dataset and the notebook that audited it have moved to the P3
repository, with the pipeline track that uses them.

### Downloaded automatically

The pretrained weights come from `download.pytorch.org` and Ultralytics, once, into a cache:

| Weights | Used in | Size | Where it goes |
|---|---|---|---|
| ImageNet-pretrained ResNet50 | `11` | about 100 MB | `~/.cache/torch/` |
| COCO-pretrained Faster R-CNN | `13` | about 160 MB | `~/.cache/torch/` |
| COCO-pretrained YOLOv8n | `13` | about 6 MB | the folder you run it from |

Each is fetched once and reused. If your network blocks `download.pytorch.org` (a school proxy, for
example), the notebooks say what to do: download elsewhere and copy the folder across. Notebook `11`
explains why silently falling back to `weights=None` invalidates a transfer-learning experiment.


Source and licence: *Weapon detection Open Data*, [ari-dasci/OD-WeaponDetection](https://github.com/ari-dasci/OD-WeaponDetection),
CC BY-SA 4.0. Cite Pérez-Hernández et al. (2020), *Knowledge-Based Systems* 194, 105590,
[doi:10.1016/j.knosys.2020.105590](https://doi.org/10.1016/j.knosys.2020.105590), in your README and your
report; the `LICENSE.md` inside the zip has the full attribution and lists what was changed. The copy you
receive is a teaching copy with a few deliberate errors in it (notebook `04`, section 12): finding them is part of the
audit.

The image assignment uses only Sohas. Do not substitute CIFAR-10, ImageNet, COCO or Pascal VOC.

## Compute

Everything runs on a CPU. The **How to use** block of each notebook states its runtime; the longest cells
are:

| Notebook | Cell | On a laptop CPU |
|---|---|---|
| `03` | training on MNIST, then on CIFAR-10 | 2–5 minutes, then 5–15 minutes |
| `07` | the training-curve experiments | about 2 minutes in total |
| `09`, `11` | the augmentation and transfer experiments | about 20 seconds each |
| `13` | one training step of a ResNet50 detector | seconds per image |

Part B of the image assignment fine-tunes a small YOLO detector, which runs at minutes per epoch on a
CPU for a few hundred images; notebook `13` times one epoch before you plan. The optional Faster R-CNN
comparison in `13` trains a ResNet50-based model and assumes a GPU.

## Installation

Install [uv](https://docs.astral.sh/uv/), clone the repository, and run — at home, before the first lesson,
because PyTorch is a large download:

```bash
uv sync
```

That creates `.venv/` with the packages in `pyproject.toml` at the versions in `uv.lock`: NumPy, pandas,
matplotlib, seaborn, scikit-learn, PyTorch, torchvision, TensorBoard, Streamlit and joblib (the last two
were for the pipeline track, now in P3, and can go at the next `uv lock`). Notebook `00` explains what those files are and how to point a notebook
kernel at the environment. TensorFlow and Keras are not used.

One extra is needed from week 6 on — YOLO for part B of the image assignment (`13`):

```bash
uv sync --extra yolo
```

This adds Ultralytics and OpenCV.

## Repository layout

```text
notebooks/      00-13
data/           datasets, downloaded or provided; ignored by git
pyproject.toml  dependencies
uv.lock         exact versions
```
