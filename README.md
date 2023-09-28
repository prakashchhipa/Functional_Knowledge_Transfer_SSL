# Title
Functional Knowledge Transfer with Self-supervised Representation Learning

# Venue
Accepted at [IEEE International Conference on Image Processing](https://2023.ieeeicip.org/) (ICIP 2023)

Chhipa, Prakash Chandra, Muskaan Chopra, Gopal Mengi, Varun Gupta, Richa Upadhyay, Meenakshi Subhash Chippa, Kanjar De, Rajkumar Saini, Seiichi Uchida, and Marcus Liwicki. "Functional Knowledge Transfer with Self-supervised Representation Learning." In 2023 IEEE International Conference on Image Processing (ICIP), pp. 3339-3343. IEEE, 2023.

# Article
[IEEE](https://ieeexplore.ieee.org/abstract/document/10222142)
[Arxiv Version](https://arxiv.org/pdf/2304.01354.pdf)

# Poster & Presentation Video 

**Click [here](https://github.com/prakashchhipa/Functional_Knowledge_Transfer_SSL/blob/main/content/Poster_Functional_Knowledge_Transfer_with_Self-supervised_Representation_Learning.pdf) for enlarged view**
<p align="center" >
  <img src="https://github.com/prakashchhipa/Functional_Knowledge_Transfer_SSL/blob/main/poster_icon.JPG" height= 70%  width= 50%>
</p>

**Short video presentation (4 minutes) describing the work**
[![IMAGE ALT TEXT HERE](https://github.com/prakashchhipa/Functional_Knowledge_Transfer_SSL/blob/main/content/video_icon_github.JPG)](https://www.youtube.com/watch?v=GlnDm_GrVm0)

# Abstract
This work investigates the unexplored usability of self-supervised representation learning in the direction of functional knowledge transfer. In this work, functional knowledge transfer is achieved by joint optimization of self-supervised learning pseudo task and supervised learning task, improving supervised learning task performance. Recent progress in self-supervised learning uses a large volume of data, which becomes a constraint for its applications on small-scale datasets. This work shares a simple yet effective joint training framework that reinforces human-supervised task learning by learning self-supervised representations just-in-time and vice versa. Experiments on three public datasets from different visual domains, Intel Image, CIFAR, and APTOS, reveal a consistent track of performance improvements on classification tasks during joint optimization. Qualitative analysis also supports the robustness of learnt representations.

# Method
[SimCLR](http://proceedings.mlr.press/v119/chen20j.html) contrastive learning method employed for self-supervised representation learning part.
<p align="center">
  <img src="https://github.com/prakashchhipa/Functional_Knowledge_Transfer_SSL/blob/main/figures/method.png">
</p>

# Datasets
Three publically available datasets from diverse visual domains are chosen for exprimentations.

1. [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html) - The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck), with 6000 images per class. There are 50000 training images and 10000 test images.
2. [Intel Images](https://www.kaggle.com/datasets/puneet6060/intel-image-classification) - This is image data of Natural Scenes around the world. This Data contains around 25000 images (of which 17034 used) of size 150x150 distributed under six categories (buildings, forest, glaciar, mountain, sea, and street).
3. [APTOS 2019](https://www.kaggle.com/competitions/aptos2019-blindness-detection/data) - A set of 3662 retina images of fundus photography under a variety of imaging conditions. A clinician has rated each image for the severity of diabetic retinopathy on a scale of 0 to 4 (0: No DR, 1: Mild, 2: Moderate, 3: Severe, and 4: Proliferative DR). 

# Results
All the expriments have batch size of 256 and ResNet50 encoder.
<p align="center">
  <img src="https://github.com/prakashchhipa/Functional_Knowledge_Transfer_SSL/blob/main/figures/results.png">
</p>

# Qualitative
<p align="center">
  <img  src="https://github.com/prakashchhipa/Functional_Knowledge_Transfer_SSL/blob/main/figures/qualitative_analysis.png">
</p>

# Model Weights

1. Functional Knowledge Transfer Jointly Trained Models
   a. [ResNet50 on CIFAR10](https://drive.google.com/file/d/1J12zUXu0v7BvqfdijdBHGYdhzOr0tzQn/view?usp=share_link)
   b. [ResNet50 on Aptos 2019](https://drive.google.com/file/d/1-Mvtb8a2i1t36nP9E3ZssqauMU4GJ7Lc/view?usp=share_link)
   c. [ResNet50 on Intel Images](https://drive.google.com/file/d/1acVhOUPz7pOLXbdA8YTFqa_yIFP58XM7/view?usp=share_link)
   
2. SSL Pretrained Models
   a. [ResNet50 on CIFAR10](https://drive.google.com/file/d/1hWCnT6Wcf_gipAra7aVNlStzgFaEa-Kk/view?usp=share_link)
   b. [ResNet50 on Aptos 2019](https://drive.google.com/file/d/1fP3kgFOlpoZ7_1roNv7e8rC6GzOTzVhl/view?usp=share_link)
   c. [ResNet50 on Intel Images](https://drive.google.com/file/d/1hWCnT6Wcf_gipAra7aVNlStzgFaEa-Kk/view?usp=share_link)

  
# Commands

1. Pretrain (for representational transfer)

```python -m pretrain <resnet_version> <device> <dataset>```

2. Finetune - downstream task

```python -m finetune train <resnet_version> <device> <dataset> <pretrained_model_weights_path>```

3. Joint training (for Functional represetation transfer)

```python -m joint_train <resnet_version> <device> <dataset>```





