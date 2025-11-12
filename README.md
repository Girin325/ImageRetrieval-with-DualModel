# Improving Image-Retrieval Performance of Foundation Models in Gastrointestinal Endoscopic Images

---


[[Project Page]](https://github.com/Girin325/ImageRetrieval-with-DualModel)

PyTorch implementation of "Improving Image-Retrieval Performance of Foundation Models in Gastrointestinal Endoscopic Images"


----

## Installation

This repository is built upon [DINOv2 (Facebook Research)]("https://github.com/facebookresearch/dinov2") and extends it into a dual-backbone image retrieval framework optimized for gastrointestinal endoscopic image analysis.

We used the __Python 3.10.__

```shell
git clone https://github.com/Girin325/ImageRetrieval-with-DualModel.git
cd ImageRetrieval-with-DualModel
conda env create -f conda.yaml
conda activate IRDM
pip install -r requirements.txt
```
---
## Data preparation
We used __Kvasir__ and __HyperKvasir__ datasets for training and evaluation.

### Dataset
- [__Kvasir__]("https://dl.acm.org/doi/abs/10.1145/3083187.3083212") – a public dataset of gastrointestinal endoscopy images containing normal and abnormal findings across eight classes.
- [__HyperKvasir__]("https://www.nature.com/articles/s41597-020-00622-y") - a large-scale extension of Kvasir with more than 110,000 images and videos covering anatomical landmarks, pathological findings, and normal variants.
