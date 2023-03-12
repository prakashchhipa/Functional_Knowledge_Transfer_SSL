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
import warnings
warnings.filterwarnings("ignore")
import os
import glob
import time
from skimage import io
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torch import nn
from torch.optim.optimizer import Optimizer, required
import sys

print("\n SSL pretrain script: ", sys.argv[0])
rn = int(sys.argv[1])
DEVICE = torch.device(f"cuda:{sys.argv[2]}")
dataset = sys.argv[3]

def get_complete_transform(output_shape, kernel_size, s=1.0):
    """
    The color distortion transform.
    
    Args:
        s: Strength parameter.
    
    Returns:
        A color distortion transform.
    """
    rnd_crop = Resize(output_shape)    # random crop
    rnd_flip = RandomHorizontalFlip(p=0.5)     # random flip
    rnd_vflip= RandomVerticalFlip(p=0.5)
    rnd_rotate = RandomRotation(degrees=(-90, 90), fill=(0,))
#     color_jitter = ColorJitter(0.01*s, 0.02*s, 0.08*s, 0.1*s)
#     rnd_color_jitter = RandomApply([color_jitter], p=0.8)      # random color jitter
    rnd_aff= RandomAffine(degrees=(-180, 180), translate=(0.2, 0.2))
#     rnd_gray = RandomGrayscale(p=0.2)    # random grayscale
    gblur =GaussianBlur(kernel_size=9, sigma=(0.1, 0.5))
#     norm= Normalize((0.125753653049469, 0.02737451672554016, 0.02293757855892181), (0.02670302987098694, 0.02240527391433716, 0.0286241775751114))
    to_tensor = ToTensor()
    image_transform = Compose([
        to_tensor,
        rnd_crop,
        rnd_flip,
        rnd_rotate,
#         rnd_gray,
        rnd_vflip,
        rnd_aff,
        gblur
#         norm
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
    
    def __init__(self, list_images, transform=None):
        self.list_images = list_images
        self.transform = transform

    def __len__(self):
        return len(self.list_images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        img_name = self.list_images[idx]
#         print(img_name)
        image = io.imread(img_name)
        if self.transform:
            image = self.transform(image)

        return image
    

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self, x):
        return x

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
#DEVICE = torch.device("cuda:03" if torch.cuda.is_available() else "cpu")
print(f'DEVICE: {DEVICE}')

output_shape = [224, 224]
kernel_size = [21,21] # 10% of the output_shape

# base SimCLR data augmentation
base_transforms = get_complete_transform(output_shape=output_shape, kernel_size=kernel_size, s=1.0)

# The custom transform
custom_transform = ContrastiveLearningViewGenerator(base_transform=base_transforms)

trainn_ds = None
validd_ds = None

if dataset == "cifar":
    # complete dataset
    trainn_ds = CustomDataset(
        list_images=glob.glob("/home/datasets/cifar10_fn/cifar/train/**/*.png",recursive = True),
        
        transform=custom_transform
    )
    validd_ds = CustomDataset(
        list_images=glob.glob("/home/datasets/cifar10_fn/cifar/test/**/*.png",recursive = True),
        transform=custom_transform
    )
elif dataset == "aptos":
     # complete dataset
    trainn_ds = CustomDataset(
        list_images=glob.glob("/home/datasets/aptos2019/train_images/train_images/*.png",recursive = True),
        
        transform=custom_transform
    )
    validd_ds = CustomDataset(
        list_images=glob.glob("/home/datasets/aptos2019/test_images/test_images/*.png",recursive = True),
        transform=custom_transform
    )
else:
     # complete dataset
    trainn_ds = CustomDataset(
        list_images=glob.glob("/home/datasets/intel_scene/seg_train/seg_train/**/*.jpg",recursive = True),
        
        transform=custom_transform
    )
    validd_ds = CustomDataset(
        list_images=glob.glob("/home/datasets/intel_scene/seg_test/seg_test/**/*.jpg",recursive = True),
        transform=custom_transform
    )

print('len(trainn_ds)', len(trainn_ds), 'len(validd_ds)', len(validd_ds))


BATCH_SIZE = 256

# Building the data loader
train_dl = torch.utils.data.DataLoader(
    trainn_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=os.cpu_count(),
    drop_last=True,
    pin_memory=True,
)


LABELS = torch.cat([torch.arange(BATCH_SIZE) for i in range(2)], dim=0)
LABELS = (LABELS.unsqueeze(0) == LABELS.unsqueeze(1)).float() # Creates a one-hot with broadcasting
LABELS = LABELS.to(DEVICE) #128,128

from torch import optim
simclr_model = SimCLR(rn = rn).to(DEVICE)       # network model
criterion = nn.CrossEntropyLoss().to(DEVICE)        # loss
# optimizer = torch.optim.Adam(simclr_model.parameters(), lr=0.79, weight_decay=1e-6)     # optimizer
# optimizer = LARS(optim.SGD(clr_model.parameters(), lr=0.59, weight_decay=1e-6))
optimizer = torch.optim.SGD(simclr_model.parameters(), lr=0.001/2, momentum=0.9, weight_decay=0.0005, nesterov=True)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000)

from tqdm import tqdm

EPOCHS = 101
checkpoint = 7
for epoch in range(EPOCHS):
    t0 = time.time()
    running_loss = 0.0
    for i, views in enumerate(tqdm(train_dl)):
        projections = simclr_model([view.to(DEVICE) for view in views])
        logits, labels = cont_loss(projections, temp=0.5)
        loss = criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    scheduler.step()
        # print statistics
    if(epoch % 10 == 0) :                        
        print(f'EPOCH: {epoch+51} BATCH: {i+1} LOSS: {(running_loss/100):.4f} ')
        
        # Checkpoint 
        torch.save(simclr_model.state_dict(), f'/home/functional_transfer_ssl/models/simclr_resnet{rn}_pretrained_two_stage_{checkpoint}_{dataset}') 
        checkpoint += 1 
    running_loss = 0.0
    print(f'Time taken: {((time.time()-t0)/60):.3f} mins')
