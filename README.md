# DLA-GAN

## Requirements
- python 3.8
- Pytorch 1.9
- At least 1x12GB NVIDIA GPU

## Installation

Clone this repo.
```
git clone https://github.com/TRICKticky/DLA-GAN.git
pip install -r requirements.txt
cd DLA-GAN/code/
```

## Experimental configuration and key algorithms: 
The experimental configuration is here:  
  - CUB: `DLA-GAN/code/cfg/bird.yml`  
  - COCO: `DLA-GAN/code/cfg/coco.yml`

The descriptions and implementations of key algorithms is here: `DLA-GAN/code/models/GAN.py`

## Preparation
### Datasets
1. Download the preprocessed metadata for [birds](https://drive.google.com/file/d/1I6ybkR7L64K8hZOraEZDuHh0cCJw5OUj/view?usp=sharing) [coco](https://drive.google.com/file/d/15Fw-gErCEArOFykW3YTnLKpRcPgI_3AB/view?usp=sharing) and extract them to `data/`
2. Download the [birds](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html) image data. Extract them to `data/birds/`
3. Download [coco2014](http://cocodataset.org/#download) dataset and extract the images to `data/coco/images/`


## Training
  ```
  cd DLA-GAN/code/
  ```
### Train the DLA-GAN model
  - For bird dataset: `bash scripts/train.sh ./cfg/bird.yml`
  - For coco dataset: `bash scripts/train.sh ./cfg/coco.yml`
### Resume training process
If your training process is interrupted unexpectedly, set **resume_epoch** and **resume_model_path** in train.sh to resume training.

### TensorBoard
Our code supports automate FID evaluation during training, the results are stored in TensorBoard files under ./logs. You can change the test interval by changing **test_interval** in the YAML file.
  - For bird dataset: `tensorboard --logdir=./code/logs/bird/train --port 8166`
  - For coco dataset: `tensorboard --logdir=./code/logs/coco/train --port 8177`

## Evaluation

### Evaluate DLA-GAN models
We synthesize about 3w images from the test descriptions and evaluate the FID between **synthesized images** and **test images** of each dataset.
  ```
  cd DLA-GAN/code/
  ```
- For bird dataset: `bash scripts/calc_fid.sh ./cfg/bird.yml`
- For coco dataset: `bash scripts/calc_fid.sh ./cfg/coco.yml`
- We compute inception score for models trained on birds using [StackGAN-inception-model](https://github.com/hanzhanggit/StackGAN-inception-model). 
- If you wanna test SSD metric, the code is here: https://github.com/zhaorui-tan/PDF-GAN_pr2023.git  
### Some tips
- Our evaluation codes do not save the synthesized images (about 3w images). If you want to save them, set **save_image: True** in the YAML file.
- Since we find that the IS can be overfitted heavily through Inception-V3 jointed training, we do not recommend the IS metric for text-to-image synthesis.

## Sampling
  ```
  cd DLA-GAN/code/
  ```
### Synthesize images from example captions
  - For bird dataset: `bash scripts/sample.sh ./cfg/bird.yml`
  - For coco dataset: `bash scripts/sample.sh ./cfg/coco.yml`
  
### Synthesize images from your text descriptions
  - Replace your text descriptions into the ./code/example_captions/dataset_name.txt
  - For bird dataset: `bash scripts/sample.sh ./cfg/bird.yml`
  - For coco dataset: `bash scripts/sample.sh ./cfg/coco.yml`

The synthesized images are saved at ./code/samples.

## Thanks for code from:  

DF-GAN:  
https://github.com/tobran/DF-GAN.git  
AttnGAN:  
https://github.com/taoxugit/AttnGAN.git  

