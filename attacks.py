from tqdm import tqdm, trange
import numpy as np
import torch                                        # root package
from torch.utils.data import Dataset, DataLoader    # dataset representation and loading
from torch import Tensor                            # tensor node in the computation graph
import torch.nn as nn                               # neural networks
import torch.nn.functional as F                     # layers, activations and more
import torch.optim as optim                         # optimizers e.g. gradient descent, ADAM, etc.
from torchvision import datasets, models, transforms     # vision datasets, architectures & transforms
import torchvision.transforms as transforms              # composable transforms

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

def FGSM(model, image, attack_label, criterion, epsilon=0.031):
    attacked_image = image.clone()
    attacked_image.requires_grad=True
    pred = model(attacked_image.to(device))
    criterion(pred, attack_label).backward()
    attacked_image.requires_grad=False
    attacked_image = attacked_image + epsilon*attacked_image.grad.sign()
    # attacked_image[attacked_image > 1.0] = 1.0
    # attacked_image[attacked_image < -1.0] = -1.0
    attacked_image.grad = None
    return attacked_image

def PGD(model, image, attack_label, criterion, epsilon=0.031, delta=2*1e-2, t_max=10, l2=False):
    attacked_image = image.clone()
    attacked_image = attacked_image.to(device)
    attack = torch.zeros_like(attacked_image).to(device)
    #print(attack_label)
    pred = model(attacked_image)
    #print(pred.max(1).indices)
    for _ in range(t_max):
        attacked_image.requires_grad = True
        pred = model(attacked_image)
        #print(pred.max(1).indices)
        #if pred.max(1).indices != attack_label:
        #    return attack
        criterion(pred, attack_label).backward()
        grad = attacked_image.grad
        attacked_image.grad = None
        attacked_image.requires_grad = False
        if l2:
            attack += delta * nn.functional.normalize(grad, p=2)
            # project back into the ball
            attack = epsilon * nn.functional.normalize(attack, p=2)
        else:
            attack += delta * grad.sign()
            # project back into the ball
            if attack.max() > epsilon:
                attack = (attack/attack.max()) * epsilon
            if -attack.min() > epsilon:
                attack = (attack/-attack.min()) * epsilon
        
        attacked_image = image + attack
        
        #attacked_image[attacked_image > 1.0] = 1.0
        #attacked_image[attacked_image < -1.0] = -1.0
    
    #pred = model(attacked_image)
    #print(pred.max(1).indices)
    return attacked_image


