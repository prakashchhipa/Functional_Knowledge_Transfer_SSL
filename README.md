# Title
Functional Knowledge Transfer with Self-supervised Representation Learning

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

# Pretrained Model



# Commands

1. Pretrain (for representational transfer)

```python -m pretrain <resnet_version> <device> <dataset>```

2. Finetune - downstream task

```python -m finetune train <resnet_version> <device> <dataset> <pretrained_model_weights_path>```

3. Joint training (for Functional represetation transfer)

```python -m joint_train <resnet_version> <device> <dataset>```





