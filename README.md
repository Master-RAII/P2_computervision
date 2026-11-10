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
| 7 | `14` Video Foundations — frames, clips, 3D convolution; section 8 (UCF101 practice) at home | `15` Video Benchmark — **the video assignment** |

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
| UCF101 subset, ten classes | `14`, section 8 | practising the video chain; `ucf101_subset.zip` | about 650 MB | `data/raw/ucf101_subset/` |
| **Sohas classification** | `04`, `11` | **image assignment, part A**; `sohas_classification.zip` | about 150 MB | `data/raw/sohas_classification/` |
| **Sohas detection** | `13` | **image assignment, part B**; `sohas_detection.zip` | about 1.8 GB | `data/raw/sohas_detection/` |
| **SCFD** (Surveillance Camera Fight Dataset) | `15` | **video assignment**; `scfd.zip` | about 90 MB | `data/raw/scfd/` |

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
| Kinetics-400-pretrained R3D-18 | `14`, `15` | about 125 MB | `~/.cache/torch/` |
| Features cached from the UCF101 subset | `14` | tens of MB | `data/ucf101_features/` (delete it when the clips change) |

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


### The video assignment: SCFD (`15`)

The video assignment uses the **Surveillance Camera Fight Dataset** (SCFD): 300 clips of about two seconds
from surveillance cameras, 150 *fight* and 150 *no fight*, from `scfd.zip`, in one folder per class:

```text
data/raw/scfd/
    fight/fi001.mp4
    no_fight/nofi001.mp4
```

The clips were cut from 44 YouTube videos and two surveillance datasets, and the file that comes with the
data says which clip came from which source. That is your recording identifier, and the first thing the
audit checks: whether the sources line up with the classes, and whether any source ends up on both sides of
your split.

Source and licence: [seymanurakti/fight-detection-surv-dataset](https://github.com/seymanurakti/fight-detection-surv-dataset),
MIT licence. Cite Aktı, Tataroğlu & Ekenel (2019), *Vision-based Fight Detection from Surveillance
Cameras*, IPTA 2019, [arXiv:2002.04355](https://arxiv.org/abs/2002.04355). SCFD is also one of the two
benchmarks of Veltmeijer, Franken & Gerritsen (2025), *Real-time violence detection and localization through
subgroup analysis*, Multimedia Tools and Applications 84(7), 3793–3807,
[doi:10.1007/s11042-024-19144-5](https://doi.org/10.1007/s11042-024-19144-5), which goes one step beyond
this assignment: it does not only classify a clip but tracks the groups of people involved. Read it for
what a fight classifier can be extended to, and for the accuracies the literature reports on SCFD.

A clip label hides three decisions — when the event starts and ends, what counted as a fight, and which
camera — and notebook `15`, section 1, asks you to watch twenty clips per class and write down the policy
you observe before training. For the optional external-validity measurement (notebook `15`, exercise 1C),
the Smart-City CCTV Violence Detection dataset (SCVD, Kaggle, CC BY-NC-SA 4.0) is a dataset of a different
source to *evaluate* on, never to train on.
Notebook `14` covers reading, sampling and splitting by recording; notebook `15` the split file every
experiment shares. Do not mix the video dataset with the image dataset, and do not compare their numbers.

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

The video part is heavier still. The synthetic experiment in `14` runs in a minute or two on a CPU,
but one clip through R3D-18, or through ResNet50 eight frames at a time, costs a second or more each, and
the benchmark in `15` multiplies that by clips, epochs, configurations and seeds. Notebook `14`, section 4,
times it on your machine and says what to cut first if it does not fit; section 8 shows how caching the
features of a frozen backbone takes epochs and configurations out of that sum.

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

Two extras are needed from week 6 on — YOLO for part B of the image assignment (`13`), and a video decoder
for `14` and `15`:

```bash
uv sync --extra yolo --extra video
```

The first adds Ultralytics and OpenCV; the second PyAV, which notebook `14`'s `load_clip` uses to decode
video frames (OpenCV is the alternative). `src/yolo_webcam.py` is a starting point for the optional webcam demonstration in `13` —
run it from a terminal, not from a notebook.

## Repository layout

```text
notebooks/      00-15
src/            yolo_webcam.py, the optional webcam demonstration of notebook 13
data/           datasets, downloaded or provided; ignored by git
pyproject.toml  dependencies
uv.lock         exact versions
```
