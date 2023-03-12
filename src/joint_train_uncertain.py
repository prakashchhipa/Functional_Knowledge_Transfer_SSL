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
    RandomAffine,
    Normalize,
    RandomGrayscale,
    RandomApply,
    Compose,
    GaussianBlur,
    ToTensor,
)
import torchvision.models as models

import os
import glob
import time
from skimage import io
import matplotlib.pyplot as plt
from pathlib import Path

from torchvision.transforms import (
    CenterCrop,
    Resize
)
import sys

print("\n SSL joint training script: ", sys.argv)
rn = int(sys.argv[1])
DEVICE = torch.device(f"cuda:{sys.argv[2]}")
dataset = sys.argv[3]

BATCH_SIZE = 256

def get_complete_transform(output_shape, kernel_size, s=1.0):
    """
    The color distortion transform.
    
    Args:
        s: Strength parameter.
    
    Returns:
        A color distortion transform.
    """
    rnd_crop = RandomResizedCrop(output_shape)    # random crop
    rnd_flip = RandomHorizontalFlip(p=0.5)     # random flip
    rnd_flip = RandomVerticalFlip(p=0.5)     # random flip
    rnd_affine = RandomAffine(degrees=(-180, 180), translate=(0.2, 0.2))
    color_jitter = ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
    rnd_color_jitter = RandomApply([color_jitter], p=0.8)      # random color jitter
    
    rnd_gray = RandomGrayscale(p=0.2)    # random grayscale
    gaussian_blur = GaussianBlur(kernel_size=kernel_size)
    rnd_gaussian_blur = RandomApply([gaussian_blur], p=0.5)
    to_tensor = ToTensor()
    norm= Normalize((0.425753653049469, 0.29737451672554016, 0.21293757855892181), 
                    (0.27670302987098694, 0.20240527391433716, 0.1686241775751114))
    image_transform = Compose([
        to_tensor,
        rnd_crop,
        rnd_flip,
        rnd_gray,
        rnd_gaussian_blur,
        rnd_affine,
        norm
    ])
    return image_transform


# generate two views for an image
class ContrastiveLearningViewGenerator(object):
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform, n_views=2):
        self.base_transform = base_transform
        self.n_views = n_views

    def __call__(self, x):
        views = [self.base_transform(x) for i in range(self.n_views)]
        return views
    
    
class CustomDataset(Dataset):
    """Flowers Dataset"""

    def __init__(self, list_images, dataset = 'cifar', transform=None):
        """
        Args:
            list_images (list): List of all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.list_images = list_images
        self.transform = transform
        self.class_transform = Compose([
            ToTensor(),
            Resize(224),
            CenterCrop(224)
            
            ])

    def __len__(self):
        return len(self.list_images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        img_name = self.list_images[idx]
        image = io.imread(img_name)
        if self.transform:
            image1 = self.transform(image)
            image2 = self.class_transform(image)
        
        if 'cifar' == dataset:    
            class_labels = ['0' , '1', '2', '3', '4', '5', '6', '7', '8', '9']
        elif 'intel' == dataset:
            class_labels = ['buildings' , 'forest', 'glacier', 'mountain', 'sea', 'street']
        else:
            class_labels = ['0' , '1', '2', '3', '4']
        
                                
        #print("Path(img_name)=", Path(img_name))
        #print("Path(img_name).parent.name=", Path(img_name).parent.name)
        #print("torch.tensor(class_labels.index(Path(img_name).parent.name)).long()=", torch.tensor(class_labels.index(Path(img_name).parent.name)).long())
        return image1, image2, torch.tensor(class_labels.index(Path(img_name).parent.name)).long()
    

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self, x):
        return x

class SimCLR(nn.Module):
    def __init__(self, rn):
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
            resnet = models.resnet101(pretrained=True)
            print("ResNet version - ", rn)
        resnet.fc = Identity()
        self.encoder = resnet
        self.projection = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 128)
        )
        self.linear = nn.Linear(512, nu_classes)
        #uncertainity implementation 
        self.log_uncertain_vars = nn.Parameter(torch.zeros((2)))
        
    def forward(self, x,linear_eval=False):
        self.linear_eval = linear_eval
        if not self.linear_eval:
            x = torch.cat(x, dim=0)
        
        encoding = self.encoder(x)
        if not self.linear_eval:
            pred = self.projection(encoding)
        else:
            pred = self.linear(encoding) 
        return pred
    
LABELS = torch.cat([torch.arange(BATCH_SIZE) for i in range(2)], dim=0)
LABELS = (LABELS.unsqueeze(0) == LABELS.unsqueeze(1)).float() # Creates a one-hot with broadcasting
LABELS = LABELS.to(DEVICE) #128,128 
# contrastive loss
def cont_loss(features, temp):
    """
    The NTxent Loss.
    
    Args:
        z1: The projection of the first branch
        z2: The projeciton of the second branch
    
    Returns:
        the NTxent loss
    """
    similarity_matrix = torch.matmul(features, features.T) # 128, 128
    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(LABELS.shape[0], dtype=torch.bool).to(DEVICE)
    # ~mask is the negative of the mask
    # the view is required to bring the matrix back to shape
    labels = LABELS[~mask].view(LABELS.shape[0], -1) # 128, 127
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1) # 128, 127

    # select and combine multiple positives
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1) # 128, 1

    # select only the negatives
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1) # 128, 126

    logits = torch.cat([positives, negatives], dim=1) # 128, 127
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(DEVICE)

    logits = logits / temp
    return logits, labels

   

print(f'Torch-Version {torch.__version__}')
print(f'DEVICE: {DEVICE}')


    
output_shape = [224,224]
kernel_size = [21,21] # 10% of the output_shape

# base SimCLR data augmentation
base_transforms = get_complete_transform(output_shape=output_shape, kernel_size=kernel_size, s=1.0)

# The custom transform
custom_transform = ContrastiveLearningViewGenerator(base_transform=base_transforms)

nu_classes = None

if "intel" == dataset:
    # complete dataset
    train_ds = CustomDataset(
        list_images=glob.glob("/home/prachh/datasets/intel_scene/seg_train/seg_train/**/*.jpg",recursive = True),
        dataset= "intel",
        transform=custom_transform
    )
    valid_ds = CustomDataset(
        list_images=glob.glob("/home/prachh/datasets/intel_scene/seg_test/seg_test/**/*.jpg",recursive = True),
        dataset= "intel",
        transform=custom_transform
    )
    nu_classes = 6
elif 'cifar' == dataset:
    train_ds = CustomDataset(
        list_images=glob.glob("/home/prachh/datasets/cifar10_fn/cifar/train/**/*.png",recursive = True),
        dataset= "cifar",
        transform=custom_transform
    )
    valid_ds = CustomDataset(
        list_images=glob.glob("/home/prachh/datasets/cifar10_fn/cifar/test/**/*.png",recursive = True),
        dataset= "cifar",
        transform=custom_transform
    )
    nu_classes = 10
else:
    train_ds = CustomDataset(
        list_images=glob.glob("/home/prachh/datasets/aptos/train/**/*.png",recursive = True),
        dataset= "aptos",
        transform=custom_transform
    )
    valid_ds = CustomDataset(
        list_images=glob.glob("/home/prachh/datasets/aptos/test/**/*.png",recursive = True),
        dataset= "aptos",
        transform=custom_transform
    ) 
    nu_classes = 5
    

# train and valid dataset

# ds = ImageFolder(
#     root="../../breakhis3/",
#     transform=custom_transform
# )




# train_size = int(0.8 * len(ds))
# valid_size = len(ds) - train_size
# train_ds, valid_ds = torch.utils.data.random_split(ds, [train_size, valid_size])

print(len(train_ds))
print(len(valid_ds))

# Building the data loader
train_dl = torch.utils.data.DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=os.cpu_count(),
    drop_last=True,
    pin_memory=True,
)

valid_dl = torch.utils.data.DataLoader(
    valid_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=os.cpu_count(),
    drop_last=True,
    pin_memory=True,
)


simclr_model = SimCLR(rn=rn).to(DEVICE)       # network model
criterion = nn.CrossEntropyLoss().to(DEVICE)        # loss
optimizer = torch.optim.SGD(simclr_model.parameters(),lr=0.00025,momentum=0.9,weight_decay=1e-6)

from tqdm import tqdm
epochs = 100
min_val_loss = 100000.0
best_val_acc = 0.0
precision_contrastive_loss_list_dir = {}
precision_classification_loss_list_dir = {}
log_uncertain_vars_list_dir = {}
for epoch in range(epochs):
    train_cont_losses = list()
    valid_cont_losses = list()
    
    train_class_losses = list()
    valid_class_losses = list()
    
    total_train_losses = list()
    total_valid_losses = list()
    
    accuracies = list()
    simclr_model.train()
    
    precision_contrastive_loss_list = []
    precision_classification_loss_list = []
    log_uncertain_vars_list = []
    
    for views , classImage, class_label in tqdm(train_dl):
        projections = simclr_model([view.to(DEVICE) for view in views])
        logits, labels = cont_loss(projections, temp=0.5)
        contrastive_loss = criterion(logits, labels)
        class_label = class_label.to(DEVICE)
        pred = simclr_model.to(DEVICE)(classImage.to(DEVICE), True)
        classification_loss = criterion(pred, class_label)
        #implementing uncertainity
        precision_contrastive_loss = torch.exp(simclr_model.log_uncertain_vars[0])
        precision_classification_loss = torch.exp(simclr_model.log_uncertain_vars[1])
        loss = torch.sum(precision_contrastive_loss * contrastive_loss + simclr_model.log_uncertain_vars[0])
        loss += torch.sum(precision_classification_loss * classification_loss + simclr_model.log_uncertain_vars[1])
        #loss = contrastive_loss + classification_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        accuracies.append(class_label.eq(pred.detach().argmax(dim =1)).float().mean())
    
        train_cont_losses.append(contrastive_loss.item())
        train_class_losses.append(classification_loss.item())
        total_train_losses.append(loss.item())
        
        precision_contrastive_loss_list.append(precision_contrastive_loss.item())
        precision_classification_loss_list.append(precision_classification_loss.item())
        #log_uncertain_vars_list.append(simclr_model.log_uncertain_vars.cpu().numpy())
  
    mean_cont_loss = torch.tensor(train_cont_losses).mean()
    mean_class_loss = torch.tensor(train_class_losses).mean()
    total_mean_loss = torch.tensor(total_train_losses).mean()
    
    precision_contrastive_loss_list_dir[epoch] = precision_contrastive_loss_list
    precision_classification_loss_list_dir[epoch] = precision_classification_loss_list
    #log_uncertain_vars_list_dir[epoch] = log_uncertain_vars_list_dir
        
    print(f'Epoch {epoch + 1}')
    print(f'total training loss: {total_mean_loss:.5f}')
    print(f'contrastive training loss: {mean_cont_loss:.5f}')
    print(f'classification training loss: {mean_class_loss:.5f}')
    print(f'classification training accuracy: {torch.tensor(accuracies).mean():.5f}')
    
    accuracies = list()
    simclr_model.eval()
    for views , classImage, class_label in tqdm(valid_dl):
        with torch.no_grad():
            projections = simclr_model([view.to(DEVICE) for view in views])
            logits, labels = cont_loss(projections, temp=0.5)
            contrastive_loss = criterion(logits, labels)

            class_label = class_label.to(DEVICE)
            pred = simclr_model.to(DEVICE)(classImage.to(DEVICE), True)
            classification_loss = criterion(pred, class_label)
            loss = contrastive_loss + classification_loss
            
        accuracies.append(class_label.eq(pred.detach().argmax(dim =1)).float().mean())
        valid_cont_losses.append(contrastive_loss.item())
        valid_class_losses.append(classification_loss.item())
        total_valid_losses.append(loss.item())
  
    mean_cont_loss = torch.tensor(valid_cont_losses).mean()
    mean_class_loss = torch.tensor(valid_class_losses).mean()
    total_mean_loss = torch.tensor(total_valid_losses).mean()
  
    print(f'Epoch {epoch + 1}')
    print(f'total validation loss: {total_mean_loss:.5f}')
    print(f'contrastive validation loss: {mean_cont_loss:.5f}')
    print(f'classification validation loss: {mean_class_loss:.5f}')
    print(f'classification validation accuracy: {torch.tensor(accuracies).mean():.5f}')
    
    curr_acc = torch.tensor(accuracies).mean()
    if best_val_acc < curr_acc:
        best_val_acc = curr_acc
        torch.save(simclr_model.state_dict(), f'/home/prachh/functional_transfer_ssl/models/epoch100_join_train_uncertain_resnet{rn}_{dataset}_{best_val_acc}')
        print(f'so far classification validation accuracy: {curr_acc:.5f}', end ='\n\n')
        print(f"saved checkpoint for epoch {epoch + 1}")

print("Best validation accuracy: ", best_val_acc)

correct = 0
total = 0
preds = []
labels = []
simclr_model.load_state_dict(torch.load(f'/home/prachh/functional_transfer_ssl/models/epoch100_join_train_uncertain_resnet{rn}_{dataset}_{best_val_acc}'))
with torch.no_grad():
    for i, (views,img,label) in enumerate(tqdm(valid_dl)):
        image = img.to(DEVICE)
        label = label.to(DEVICE)
        pred = simclr_model.to(DEVICE)(img.to(DEVICE), True)
        _, predicted = torch.max(pred.data, 1)
        preds += predicted.cpu().numpy().tolist()
        labels += label.cpu().numpy().tolist()
        total += label.size(0)
        correct += (predicted == label).sum().item()

print(f'Accuracy: {100 * correct / total} %')

import json
# create json object from dictionary
json = json.dumps(precision_contrastive_loss_list_dir)
f = open(f"/home/prachh/functional_transfer_ssl/models/resnet{rn}_{dataset}_precision1_dict.json","w")
f.write(json)
f.close()
json = json.dumps(precision_classification_loss_list_dir)
f = open(f"/home/prachh/functional_transfer_ssl/models/resnet{rn}_{dataset}_precision2_dict.json","w")
f.write(json)
f.close()
#json = json.dumps(log_uncertain_vars_list_dir)
#f = open(f"/home/prachh/functional_transfer_ssl/models/resnet{rn}_{dataset}_logvars_dict.json","w")
#f.write(json)
#f.close()
