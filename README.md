# CNN-vs-Transformers-for-UAV-Imagery
A coursework project for AI702.

> [**Fine-Grained Object Detection in Drone Imagery: A Comparative Study of Convolutional Neural Networks and Transformer-based Models**](https://mbzuaiac-my.sharepoint.com/:p:/g/personal/umair_nawaz_mbzuai_ac_ae/EWAz7BYnyqtKoSxYl2EDbDcBypvRgZR3NzelFHLQ-RfWSw?e=7lYAx0)<br>
> [Umair Nawaz](https://scholar.google.com/citations?user=w7N4wSYAAAAJ&hl=en), 
[Tooba Tehreem Sheikh](https://github.com/toobatehreem) and
[Ufaq Khan](https://scholar.google.com/citations?user=E0p-7JEAAAAJ&hl=en&oi=ao)


[![paper](https://img.shields.io/badge/Final-Report-<COLOR>.svg)](https://mbzuaiac-my.sharepoint.com/:b:/g/personal/umair_nawaz_mbzuai_ac_ae/EbSSy2q0ygFAv4TGgZ0G0ooBBNGLzps43YDh8naqc6wYxg?e=O0HQG1)
<!-- [![video](https://img.shields.io/badge/Presentation-Video-F9D371)](https://github.com/asif-hanif/media/blob/main/miccai2023/VAFA_MICCAI2023_VIDEO.mp4)
[![slides](https://img.shields.io/badge/Presentation-Slides-B762C1)](https://github.com/asif-hanif/media/blob/main/miccai2023/VAFA_MICCAI2023_SLIDES.pdf)
[![poster](https://img.shields.io/badge/Presentation-Poster-blue)](https://github.com/asif-hanif/media/blob/main/miccai2023/VAFA_MICCAI2023_POSTER.pdf) -->



<hr />
 
| ![main figure](/Media/sample.gif)|
|:--| 
<!-- | **Overview of Adversarial Frequency Attack and Training**: A model trained on voxel-domain adversarial attacks is vulnerable to frequency-domain adversarial attacks. In our proposed adversarial training method, we generate adversarial samples by perturbing their frequency-domain representation using a novel module named "Frequency Perturbation". The model is then updated while minimizing the dice loss on clean and adversarially perturbed images. Furthermore, we propose a frequency consistency loss to improve the model performance. | -->


> **Abstract:** <p style="text-align: justify;">*Advancements in drone technology have significantly enhanced the capabilities for aerial data acquisition, making the precise detection of small-scale objects a critical research area. Unlike conventional imaging techniques, drone imagery presents unique challenges due to the small size and high density of objects against varied backgrounds. This study systematically compares the performance of Convolutional Neural Networks (CNNs) and Transformer-based models in the context of fine-grained object detection in drone imagery. Our investigation employs a structured experimental methodology, utilizing the VisDrone and UAVDT datasets to assess the models across various performance metrics, including mean average precision (mAP) at multiple intersections over Union (IoU) thresholds. The comparative analysis aims to uncover each model type's relative strengths and weaknesses, focusing on their adaptability to the complexities of aerial image processing. The findings of this study are intended to inform the development of more effective object detection systems for UAVs, offering insights that are pivotal for optimizing neural network models tailored to the nuances of drone-based surveillance and analysis. This research enhances our understanding of model performance in drone imagery and sets the stage for future advancements in UAV imaging technology.* </p>
<hr />



## Updates :rocket:
- **May 01, 2024** : Submitted final report of AI Project. 
<!-- - **July 10, 2023** : Released code for attacking [UNETR](https://openaccess.thecvf.com/content/WACV2022/papers/Hatamizadeh_UNETR_Transformers_for_3D_Medical_Image_Segmentation_WACV_2022_paper.pdf) model with support for [Synapse](https://www.synapse.org/#!Synapse:syn3193805/wiki/217789) dataset.
- **May 25, 2023** : Early acceptance in [MICCAI 2023](https://conferences.miccai.org/2023/en/) (top 14%) &nbsp;&nbsp; :confetti_ball: -->


<br>

## Installation :wrench:

The model depends on the following libraries:
1. sklearn
2. PIL
3. Python >= 3.8
4. ivtmetrics
5. Developer's framework:
    1. For Tensorflow version 1:
        * TF >= 1.10
    2. For Tensorflow version 2:
        * TF >= 2.1
    3. For PyTorch version:
        - Pyorch >= 1.9.0
        - cuda == 11.1
        - TorchVision >= 0.10
    4. For MMDetection:
        - mmcv >= 2.0.0

Steps to install dependencies
1. Load necessary modules
```shell
module load gcc-7
module load cuda-11.1
```
2. Create conda environment
```shell
conda create --name mmdetection python=3.8 -y

conda activate aiproject
```
2. Install PyTorch
```shell
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
```
3. Install all other dependencies
```shell
pip install -r requirements.txt
```

## Adversarial Attacks
Code of our AI-Project can be accessed [here](attacks/).

## Dataset
<!-- We conducted experiments on two volumetric medical image segmentation datasets: [Synapse](https://www.synapse.org/#!Synapse:syn3193805/wiki/217789), [ACDC](https://www.creatis.insa-lyon.fr/Challenge/acdc/databases.html). Synapse contains 14 classes (including background) and ACDC contains 4 classes (including background). We follow the same dataset preprocessing as in [nnFormer](https://github.com/282857341/nnFormer).  -->

The dataset folders for VisDrone and UAVDT should be organized as follows: 

### VisDrone Dataset Structure

The VisDrone dataset is structured to support various computer vision tasks such as object detection, tracking, and segmentation in aerial images. Here’s the typical organization:

- **Annotations/:** Contains annotation files for each image, detailing object boundaries and class labels.
  - `train_annotations/` - Annotation files for training images.
  - `val_annotations/` - Annotation files for validation images.
  - `test_annotations/` - Annotation files for test images.

- **Images/:** This directory holds all the image files corresponding to the annotations.
  - `train_images/` - Training image files.
  - `val_images/` - Validation image files.
  - `test_images/` - Test image files.

- **README.txt:** Provides additional information about the dataset, including data collection methods, annotation details, and usage guidelines.

### UAVDT Dataset Structure

The UAVDT dataset, designed for UAV-based object detection and tracking, includes several features to facilitate advanced computer vision research:
  
- **Annotations/:** This folder contains ground truth files that include details like bounding box coordinates and object classes.
  - `train_annotations/` - Annotation files used for training models.
  - `val_annotations/` - Annotation files used for validating model accuracy.
  - `test_annotations/` - Annotation files used for testing models.

- **Images/:** Contains the actual images used for object detection and tracking.
  - `train_images/` - Images for training.
  - `val_images/` - Images for validation.
  - `test_images/` - Images for testing.

- **Benchmark/:** Often includes benchmark scripts or pre-computed results to help compare the performance of different algorithms on the dataset.

- **README.txt:** Detailed description of the dataset, including the setup, challenges, and instructions for use.

Both datasets are rich in aerial imagery and are accompanied by comprehensive annotations that provide extensive data for training and evaluating object detection models. Each dataset comes with specific instructions in their respective README files, which should be read carefully to understand how to correctly use and cite the data. Include these details in your GitHub repo to guide users on how to navigate and utilize these datasets effectively.
<br>



<br />

| Dataset | Link |
|:-- |:-- |
| CholecT45 | [Download](https://jstrieb.github.io/link-lock/#eyJ2IjoiMC4wLjEiLCJlIjoibm0veTR2L3BjVTMvWkVaamZvR0V3SFNYL2NJYzgzS1crdnp5VGtXMW8rZE4vMjBlL0J1ZUNrSVRwWmdhQUpTQi9wQlY0L3E0c25Wb25kQ3U4S1dycUxKVUtrQStYZjRVS0Y2VmY4ZnVkNysvSFpZPSIsImgiOiJQYXNzd29yZCBzZW50IGJ5IGVtYWlsIHRvIGdyYW50IGFjY2VzcyB0byBDaG9sZWNUNDUgRGF0YXNldCIsInMiOiJIWHI5dm1aVlJFYVRGQlh3b0hHMWR3PT0iLCJpIjoiUitOb3Bmd2ZseFpOQWFvdCJ9)   Password: ct45_camma_@dwxr+p|

<!-- You can use the command `tar -xzf btcv-synapse.tar.gz` to un-compress the file. -->

</br>


# Running the Model

The code can be run in a trianing mode (`-t`) or testing mode (`-e`)  or both (`-t -e`) if you want to evaluate at the end of training :

<br />

## Training on CholecT45 Dataset

Simple training on CholecT45 dataset:

```
python run.py -t  --data_dir="/path/to/dataset" --dataset_variant=cholect45-crossval --version=1
```

You can include more details such as epoch, batch size, cross-validation and evaluation fold, weight initialization, learning rates for all subtasks, etc.:

```
python3 run.py -t -e  --data_dir="/path/to/dataset" --dataset_variant=cholect45-crossval --kfold=1 --epochs=180 --batch=64 --version=2 -l 1e-2 1e-3 1e-4 --pretrain_dir='path/to/imagenet/weights'
```

All the flags can been seen in the `run.py` file.
The experimental setup of the published model is contained in the paper.

<br />

## Testing

```
python3 run.py -e --data_dir="/path/to/dataset" --dataset_variant=cholect45-crossval --kfold 1 --batch 32 --version=1 --test_ckpt="/path/to/model-k3/weights"
```

<br />



<!-- ## Model
We use [UNETR](https://openaccess.thecvf.com/content/WACV2022/papers/Hatamizadeh_UNETR_Transformers_for_3D_Medical_Image_Segmentation_WACV_2022_paper.pdf) model with following parameters:
```python
model = UNETR(
    in_channels=1,
    out_channels=14,
    img_size=(96,96,96),
    feature_size=16,
    hidden_size=768,
    mlp_dim=3072,
    num_heads=12,
    pos_embed="perceptron",
    norm_name="instance",
    conv_block=True,
    res_block=True,
    dropout_rate=0.0)

```

We also used [UNETR++](https://arxiv.org/abs/2212.04497) in our experiments but its code is not in a presentable form. Therefore, we are not including support for UNETR++ model in this repository. 

Clean and adversarially trained (under VAFA attack) [UNETR](https://openaccess.thecvf.com/content/WACV2022/papers/Hatamizadeh_UNETR_Transformers_for_3D_Medical_Image_Segmentation_WACV_2022_paper.pdf) models can be downloaded from the links given below. Place these models in a directory and give full path of the model (including name of the model e.g. `/folder_a/folder_b/model.pt`) in argument `--pretrained_path` to attack that model. -->

| Dataset | Model | Link |
|:-- |:-- |:-- | 
|CholecT45 Cross-Val | Rendezvous $(\mathcal{M})$ | [Download](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/umair_nawaz_mbzuai_ac_ae/EYfGsmktjUBKqBS5ZVzItEEBTBWcEBJGciQ388uwLL-oTw?e=lA7DKE)|



## Launch Attacks on the Model
After training the model, each attack can be launched on the model by initializing the hyper-parameters in each individual attack notebook located [here](attacks/).
<!-- ```shell
Run 
```
If adversarial images are not intended to be saved, use `--debugging` argument. If `--use_ssim_loss` is not mentioned, SSIM loss will not be used in the adversarial objective (Eq. 2). If adversarial versions of train images are inteded to be generated, mention argument `--gen_train_adv_mode` instead of `--gen_val_adv_mode`.

For VAFA attack on each 2D slice of volumetric image, use : `--attack_name vafa-2d --q_max 20 --steps 20 --block_size 32 32 --use_ssim_loss`

Use following arguments when launching pixel/voxel domain attacks:

[PGD](https://adversarial-attacks-pytorch.readthedocs.io/en/latest/attacks.html#module-torchattacks.attacks.pgd):&nbsp;&nbsp;&nbsp;        `--attack_name pgd --steps 20 --eps 4 --alpha 0.01`

[FGSM](https://adversarial-attacks-pytorch.readthedocs.io/en/latest/attacks.html#module-torchattacks.attacks.fgsm):             `--attack_name fgsm --steps 20 --eps 4 --alpha 0.01`

[BIM](https://adversarial-attacks-pytorch.readthedocs.io/en/latest/attacks.html#module-torchattacks.attacks.bim):&nbsp;&nbsp;&nbsp;        `--attack_name bim --steps 20 --eps 4 --alpha 0.01`

[GN](https://adversarial-attacks-pytorch.readthedocs.io/en/latest/attacks.html#module-torchattacks.attacks.gn):&nbsp;&nbsp;&nbsp;&nbsp;   `--attack_name gn --steps 20 --eps 4 --alpha 0.01 --std 4`

## Launch Adversarial Training (VAFT) of the Model
```shell
python run_normal_or_adv_training.py --model_name unet-r --in_channels 1 --out_channel 14 --feature_size=16 --batch_size=3 --max_epochs 5000 --optim_lr=1e-4 --lrschedule=warmup_cosine --infer_overlap=0.5 \
--save_checkpoint \
--dataset btcv --data_dir=<PATH_OF_DATASET> \
--json_list=dataset_synapse_18_12.json \
--use_pretrained \
--pretrained_path=<PATH_OF_PRETRAINED_MODEL>  \
--save_model_dir=<PATH_TO_SAVE_ADVERSARIALLY_TRAINED_MODEL> \
--val_every 15 \
--adv_training_mode --freq_reg_mode \
--attack_name vafa-3d --q_max 20 --steps 20 --block_size 32 32 32 --use_ssim_loss 
```

Arugument `--adv_training_mode` in conjunction with `--freq_reg_mode` performs adversarial training with dice loss on clean images, adversarial images and frequency regularization term (Eq. 4) in the objective function (Eq. 3). For vanilla adversarial training (i.e. dice loss on adversarial images), use only `--adv_training_mode`. For normal training of the model, do not mention these two arguments. 


## Inference on the Model with already saved Adversarial Images
If adversarial images have already been saved and one wants to do inference on the model using saved adversarial images, use following command:

```shell
python inference_on_saved_adv_samples.py --model_name unet-r --in_channels 1 --out_channel 14 --feature_size=16 --infer_overlap=0.5 \
--dataset btcv --data_dir=<PATH_OF_DATASET> \
--json_list=dataset_synapse_18_12.json \
--use_pretrained \
--pretrained_path=<PATH_OF_PRETRAINED_MODEL>  \
--adv_images_dir=<PATH_OF_SAVED_ADVERSARIAL_IMAGES> \ 
--attack_name vafa-3d --q_max 20 --steps 20 --block_size 32 32 32 --use_ssim_loss 
```

Attack related arguments are used to automatically find the sub-folder containing adversarial images. Sub-folder should be present in parent folder path specified by `--adv_images_dir` argument.  If `--no_sub_dir_adv_images` is mentioned, sub-folder will not be searched and images are assumed to be present directly in the parent folder path specified by `--adv_images_dir` argument. Structure of dataset folder should be same as specified in [Datatset](##dataset) section. -->


<!-- ## Citation
If you find our work, this repository, or pretrained models useful, please consider giving a star :star: and citation.
```bibtex
@inproceedings{hanif2023frequency,
  title={Frequency Domain Adversarial Training for Robust Volumetric Medical Segmentation},
  author={Hanif, Asif and Naseer, Muzammal and Khan, Salman and Shah, Mubarak and Khan, Fahad Shahbaz},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={457--467},
  year={2023},
  organization={Springer}
}
```

<hr /> -->

## Contact
Should you have any question, please create an issue on this repository or contact us at **umair.nawaz@mbzuai.ac.ae**, **tooba.sheikh@mbzuai.ac.ae** and **ufaq.khan@mbzuai.ac.ae**

<hr />

<!---
## Our Related Works
  --->