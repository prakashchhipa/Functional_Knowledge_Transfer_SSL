import torch
import torch.nn as nn
import torchvision

from torch.utils.data import (
    Dataset,
    DataLoader,
)

from torchvision.datasets import ImageFolder
from torchvision.transforms import (
    RandomResizedCrop,
    RandomHorizontalFlip,
    RandomVerticalFlip,
    ColorJitter,
    RandomGrayscale,
    RandomApply,
    Compose,
    GaussianBlur,
    Resize,
    ToTensor,
    RandomRotation,
    RandomAffine,
    Normalize
)
# from torchlars import LARS
import torchvision.models as models

import os
import glob
import time
from skimage import io
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torch import nn
from torch.optim.optimizer import Optimizer, required


from torchvision.transforms import (
    CenterCrop,
    Resize
)

import pandas as pd
from sklearn.preprocessing import MinMaxScaler    
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, cohen_kappa_score

import numpy as np
from sklearn.metrics import roc_curve, auc

import sys


print("\n SSL pretrain script: ", sys.argv)
flag = sys.argv[1]
rn = int(sys.argv[2])
DEVICE = torch.device(f"cuda:{sys.argv[3]}")
dataset = sys.argv[4]
model_path = sys.argv[5]
BATCH_SIZE = 256

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self, x):
        return x

class LARS(Optimizer): 

    def __init__(
        self,
        params,
        lr=required,
        momentum: float = 0,
        dampening: float = 0,
        weight_decay: float = 0,
        nesterov: bool = False,
        trust_coefficient: float = 0.001,
        eps: float = 1e-8,
    ):
        if lr is not required and lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening, weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")

        self.eps = eps
        self.trust_coefficient = trust_coefficient

        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)

        for group in self.param_groups:
            group.setdefault("nesterov", False)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # exclude scaling for params with 0 weight decay
        for group in self.param_groups:
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            dampening = group["dampening"]
            nesterov = group["nesterov"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                d_p = p.grad
                p_norm = torch.norm(p.data)
                g_norm = torch.norm(p.grad.data)

                # lars scaling + weight decay part
                if weight_decay != 0:
                    if p_norm != 0 and g_norm != 0:
                        lars_lr = p_norm / (g_norm + p_norm * weight_decay + self.eps)
                        lars_lr *= self.trust_coefficient

                        d_p = d_p.add(p, alpha=weight_decay)
                        d_p *= lars_lr

                # sgd part
                if momentum != 0:
                    param_state = self.state[p]
                    if "momentum_buffer" not in param_state:
                        buf = param_state["momentum_buffer"] = torch.clone(d_p).detach()
                    else:
                        buf = param_state["momentum_buffer"]
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                    if nesterov:
                        d_p = d_p.add(buf, alpha=momentum)
                    else:
                        d_p = buf

                p.add_(d_p, alpha=-group["lr"])

        return loss
# simclr network
class SimCLR(nn.Module):
    def __init__(self, rn = 18, linear_eval=False):
        super().__init__()
        self.linear_eval = None
        resnet = None
        if rn == 18:
            resnet = models.resnet18(pretrained=True)
            print("ResNet version - ", rn)
        elif rn == 50:
            resnet = models.resnet50(pretrained=True)
            print("ResNet version - ", rn)
        elif rn == 34:
            resnet = models.resnet34(pretrained=True)
            print("ResNet version - ", rn)
        resnet.fc = Identity()
        self.encoder = resnet
        self.projection = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 128)
        )
    def forward(self, x):
        if not self.linear_eval:
            x = torch.cat(x, dim=0)
        
        encoding = self.encoder(x)
        projection = self.projection(encoding) 
        return projection
   
# classifier    
class LinearEvaluation(nn.Module):
    def __init__(self, model, nu_classes):
        super().__init__()
        simclr = model
        simclr.linear_eval = True
        simclr.projection = Identity()
        self.simclr = simclr
        for param in self.simclr.parameters():
            param.requires_grad = True
        self.linear = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, nu_classes))
    def forward(self, x):
        encoding = self.simclr(x)
        pred = self.linear(encoding) 
        return pred

# transforms
resize = Resize(255)
ccrop = CenterCrop(224)
ttensor = ToTensor()

custom_transform = Compose([
    resize,
    ccrop,
    ttensor,
])
val_resize = Resize(224)
val_transform = Compose([
    val_resize,
    ttensor,
])
train2_ds = None
valid2_ds = None
test2_ds = None
if dataset == "Intel":
    #Intel
    train2_ds = ImageFolder(
        root="/home/datasets/intel_scene/seg_train/seg_train/",
        transform=custom_transform
    )
    valid2_ds = ImageFolder(
        root="/home/datasets/intel_scene/seg_test/seg_test/",
        transform=custom_transform
    )
    test2_ds = ImageFolder(
        root="/home/datasets/intel_scene/seg_test/seg_test/",
        transform=custom_transform
    )
    print("dataset examples - ", len(train2_ds), len(valid2_ds), len(test2_ds))
    nu_classes = 6
if dataset == "cifar":
    #Intel
    train2_ds = ImageFolder(
        root="/home/datasets/cifar10_fn/cifar/train/",
        transform=custom_transform
    )
    valid2_ds = ImageFolder(
        root="/home/datasets/cifar10_fn/cifar/test/",
        transform=custom_transform
    )
    test2_ds = ImageFolder(
        root="/home/datasets/cifar10_fn/cifar/test/",
        transform=custom_transform
    )
    print("dataset examples - ", len(train2_ds), len(valid2_ds), len(test2_ds))
    nu_classes = 10
if dataset == "aptos":
    #Intel
    train2_ds = ImageFolder(
        root="/home/datasets/aptos/train/",
        transform=custom_transform
    )
    valid2_ds = ImageFolder(
        root="/home/datasets/aptos/test/",
        transform=custom_transform
    )
    test2_ds = ImageFolder(
        root="/home/datasets/aptos/test/",
        transform=custom_transform
    )
    print("dataset examples - ", len(train2_ds), len(valid2_ds), len(test2_ds))
    nu_classes = 5
    
    


# Building the data loader
train_dl = torch.utils.data.DataLoader(
    train2_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=os.cpu_count(),
    drop_last=True,
    pin_memory=True,
)

valid_dl = torch.utils.data.DataLoader(
    valid2_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=os.cpu_count(),
    drop_last=True,
    pin_memory=True,
)

test_dl = torch.utils.data.DataLoader(
    test2_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=os.cpu_count(),
    drop_last=True,
    pin_memory=True,
)




if flag == "train":
    simclr_model = SimCLR(rn = rn).to(DEVICE)
    simclr_model.load_state_dict(torch.load(model_path)) #load_tar
    eval_model = LinearEvaluation(simclr_model, nu_classes).to(DEVICE)
    ct = 0
    for child in eval_model.children():
        ct += 1
        if ct < 2:
            for param in child.parameters():
                param.requires_grad = True
            
    
    criterion = nn.CrossEntropyLoss().to(DEVICE)
    optimizer = LARS(eval_model.parameters(), lr=0.005, momentum=0.9,  weight_decay=1e-6, nesterov=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000)
    
    train_loss = list()
    train_acc= list()
    val_loss= list()
    val_acc= list()
    from tqdm import tqdm
    epochs = 100
    min_val_loss = 100000.0
    best_val_acc = 0.0
    for epoch in range(epochs):
        accuracies = list()
        class_losses = list()
        eval_model.train()
        for class_batch in tqdm(train_dl):
            x, y = class_batch
            x = x.to(DEVICE)
            y = y.to(DEVICE)
        
            logit = eval_model(x)
            classification_loss = criterion(logit, y)
            class_losses.append(classification_loss.item())

            optimizer.zero_grad()
            classification_loss.backward()
            optimizer.step()
            accuracies.append(y.eq(logit.detach().argmax(dim =1)).float().mean())
        scheduler.step()  
        
        train_loss.append(class_losses)
        train_acc.append(accuracies)
        print(f'Epoch {epoch + 1}')
        print(f'classification training loss: {torch.tensor(class_losses).mean():.5f}')
        print(f'classification training accuracy: {torch.tensor(accuracies).mean():.5f}', 
            end ='\n\n')
        

        losses = list()
        accuracies = list()
        eval_model.eval()
        for batch in tqdm(valid_dl):
            x, y = batch
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            with torch.no_grad():
                logit =eval_model(x)

            loss = criterion(logit, y)

            losses.append(loss.item())
            accuracies.append(y.eq(logit.detach().argmax(dim =1)).float().mean())
        
        val_loss.append(losses)
        val_acc.append(accuracies)
        print(f'Epoch {epoch + 1}')
        print(f'classification validation loss: {torch.tensor(losses).mean():.5f}')
        print(f'classification validation accuracy: {torch.tensor(accuracies).mean():.5f}', 
            end ='\n\n')
        
        if min_val_loss > torch.tensor(losses).mean():
            min_val_loss = torch.tensor(losses).mean()
            best_val_acc = torch.tensor(accuracies).mean()
            torch.save(eval_model.state_dict(), f'/home/functional_transfer_ssl/models/downstream_resnet{rn}_{dataset}')
            print(f'so far classification validation loss: {torch.tensor(losses).mean():.5f}')
            print(f'so far classification validation accuracy: {torch.tensor(accuracies).mean():.5f}', 
            end ='\n\n')
            print(f"saved checkpoint for epoch {epoch + 1}")
    
    print("Best validation accuracy: ", best_val_acc)
    
if flag == "eval":
    
    simclr_model = SimCLR(rn = rn).to(DEVICE)
    eval_model = LinearEvaluation(simclr_model, nu_classes).to(DEVICE)
    eval_model.load_state_dict(torch.load(f'/home/functional_transfer_ssl/models/downstream_resnet{rn}_{dataset}')) #load_tar
    from tqdm import tqdm
    correct = 0
    total = 0
    preds = []
    labels = []
    with torch.no_grad():
        for i, element in enumerate(tqdm(test_dl)):
            image, label = element
            image = image.to(DEVICE)
            label = label.to(DEVICE)
            outputs = eval_model(image)
            _, predicted = torch.max(outputs.data, 1)
            preds += predicted.cpu().numpy().tolist()
            labels += label.cpu().numpy().tolist()
            total += label.size(0)
            correct += (predicted == label).sum().item()

    print(f'Accuracy: {100 * correct / total} %')
    
    confusion_matrix_df = pd.DataFrame(confusion_matrix(labels, preds))
    
    target_names = ['res', 'Nonres']
    print(classification_report(labels, preds, target_names=target_names))
    print(cohen_kappa_score(labels, preds, weights='quadratic'))
    
    probs = np.exp(labels[:])
    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(labels, preds)

    # Compute ROC area
    roc_auc = auc(fpr, tpr)
    print('ROC area is {0}'.format(roc_auc))
    #print(auc(fpr,tpr
    
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', label='ROC curve (area = %0.8f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.01])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.savefig(f"/home/functional_transfer_ssl/finetuned_resnet{rn}_{dataset}_.png")

    
    
    
    
