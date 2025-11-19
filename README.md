# Improving Image-Retrieval Performance of Foundation Models in Gastrointestinal Endoscopic Images



[[Project Page]](https://github.com/Girin325/ImageRetrieval-with-DualModel)

PyTorch implementation of "Improving Image-Retrieval Performance of Foundation Models in Gastrointestinal Endoscopic Images"




## Installation

This repository is built upon [DINOv2 (Facebook Research)](https://github.com/facebookresearch/dinov2) and extends it into a dual-backbone image retrieval framework optimized for gastrointestinal endoscopic image analysis.

We used the __Python 3.10.__

```shell
git clone https://github.com/Girin325/ImageRetrieval-with-DualModel.git
cd ImageRetrieval-with-DualModel
conda env create -f conda.yaml
conda activate IRDM
pip install -r requirements.txt
```

## Data preparation

We used __Kvasir__ and __HyperKvasir__ datasets for training and evaluation, as well as the __GastroHUN__ dataset for evaluation.

### Dataset
- [__Kvasir__](https://dl.acm.org/doi/abs/10.1145/3083187.3083212) – a public dataset of gastrointestinal endoscopy images containing normal and abnormal findings across eight classes.
- [__HyperKvasir__](https://www.nature.com/articles/s41597-020-00622-y) - a large-scale extension of Kvasir with more than 110,000 images and videos covering anatomical landmarks, pathological findings, and normal variants.
- [__GastroHUN__](https://www.nature.com/articles/s41597-025-04401-5) - gastrointestinal endoscopy video dataset (anatomical landmarks / findings).

## Pretrained Weights

### DINOv2
You can check and download the checkpoint for DINOv2 [here](https://github.com/facebookresearch/dinov2). 

### GastroNet
You can check and download the checkpoint for GastroNet [here](https://huggingface.co/tgwboers/GastroNet-5M_Pretrained_Weights).

Then create a folder named ```pretrained_weights``` in this lepo and place the downloaded model in it.

## Utilization

### Traiing
```shell
python train_dual_model.py --mode tri --train_dir path/to/train --val_dir path/to/val --save_path path/to/save/model.pth
```

### Test
```shell
python eval_dual_model.py --ckpt results/best_model.pth --query_dir path/to/test_query --db_dir path/to/test_database
```

### Inference
```shell
python inference.py --query_dir path/to/query_directory --db_root path/to/database_directory --checkpoint path/to/model.pth
```

## Acknowledgment

Our code is based on the implementation of [DINOv2](https://github.com/facebookresearch/dinov2), [GastroNet](https://huggingface.co/tgwboers/GastroNet-5M_Pretrained_Weights). We thank their excellent works.
