# Title
Functional Knowledge Transfer with Self-supervised Representation Learning

# Abstract
This work investigates the unexplored usability of self-supervised representation learning in the direction of functional knowledge transfer. In this work, functional knowledge transfer is achieved by joint optimization of self-supervised learning pseudo task and supervised learning task, improving supervised learning task performance. Recent progress in self-supervised learning uses a large volume of data, which becomes a constraint for its applications on small-scale datasets. This work shares a simple yet effective joint training framework that reinforces human-supervised task learning by learning self-supervised representations just-in-time and vice versa. Experiments on three public datasets from different visual domains, Intel Image, CIFAR, and APTOS, reveal a consistent track of performance improvements on classification tasks during joint optimization. Qualitative analysis also supports the robustness of learnt representations.

# Method
![alt text](https://github.com/prakashchhipa/Functional_Knowledge_Transfer_SSL/blob/main/figures/method.png)

# Results
<p align="left">
  <img width="460" height="350" src="https://github.com/prakashchhipa/Functional_Knowledge_Transfer_SSL/blob/main/figures/results.png">
</p>
<p align="right">
  <img width="460" height="350" src="https://github.com/prakashchhipa/Functional_Knowledge_Transfer_SSL/blob/main/figures/qualitative_analysis.png">
</p>

# Commands

1. Pretrain (for representational transfer)

```python -m pretrain <resnet_version> <device> <dataset>```

2. Finetune - downstream task

```python -m finetune train <resnet_version> <device> <dataset> <pretrained_model_weights_path>```

3. Joint training (for Functional represetation transfer)

```python -m joint_train <resnet_version> <device> <dataset>```





