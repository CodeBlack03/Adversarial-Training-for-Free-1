import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from tqdm import tqdm
import random
import numpy as np
from model.wideN import *
from utils import *

EPOCHS,BATCH_SIZE,ITERATION,EPSILON,STEP_SIZE,SEED,DEVICE,M,LR,MOMENT,WEIGHT_DECAY = init_args()
random.seed(SEED)
torch.cuda.manual_seed(SEED) #Set seed for cuda
torch.manual_seed(SEED) #Set seed for cpu
np.random.seed(SEED) #Set seed for NumPy
torch.backends.cudnn.deterministic = True #Set deterministic behaviour of cuda
torch.backends.cudnn.benchmark = True #allow PyTorch to optimize GPU performance dynamically based on the input size 

trainLoader,testLoader = import_data(BATCH_SIZE)
net34 = WideResNet_34_10()
delta = torch.zeros(BATCH_SIZE, 3, 32, 32)
delta = delta.to(DEVICE)
net34 = net34.to(DEVICE)

#Allow model to parallely use GPU's
if(DEVICE=="cuda" and torch.cuda.device_count() > 1):
    net34 = nn.DataParallel(net34, list(range(torch.cuda.device_count())))

optimizer = optim.SGD(net34.parameters(),lr=LR,momentum=MOMENT,weight_decay=WEIGHT_DECAY)


def train():
    accuracy_per_epoch=[]
    C_loss = nn.CrossEntropyLoss()
    #Initialise Optimizer as Stochastic Gradient Descent


    for epoch in (range(EPOCHS//M)):
        net34.train()
        global delta
        iterator = tqdm(trainLoader,ncols=0)
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.1 if epoch<12 else (0.01 if (epoch >= 12 and epoch < 22) else 0.001)
        for batch_idx,(x_train, y_train) in enumerate(iterator):
            x_train = x_train.to(DEVICE)
            y_train = y_train.to(DEVICE)
            for _ in range(M):
                optimizer.zero_grad()

                x_adv = (x_train+delta).detach()
                x_adv.requires_grad_()
                y_pred = net34(x_adv)
                
                loss = C_loss(y_pred,y_train)
                loss.backward()
                optimizer.step() #theta = theta - lr*G(theta)
                grad = x_adv.grad.data
                delta = delta.detach() + EPSILON*torch.sign(grad.detach())
                delta = torch.clamp(delta,-EPSILON,EPSILON)
                

        net34.eval()
        accuracy_per_epoch.append(float(accuracy(model, test_loader, device)))
        print(f'Accuracy at epoch {epoch}: {accuracy_per_epoch[-1]}')


if __name__ == '__main__':
    train()