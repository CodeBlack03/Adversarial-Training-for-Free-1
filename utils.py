
import torch
import torchvision
import torchvision.transforms as transforms


def import_data(batch_size):
    transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),

    ])
    transform_test = transforms.Compose([
    transforms.ToTensor(),
    ])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, drop_last=True)
    testloader = torch.utils.data.DataLoader(testset,batch_size=batch_size, shuffle=False,num_workers=2)
    return trainloader,testloader


def init_args():
    EPOCH=27
    BATCH_SIZE=128
    ITERATION=100
    EPSILON=8.0/255
    STEP_SIZE=2./255
    SEED=99
    M=8
    LR=0.1
    MOMENT=0.9
    WEIGHT_DECAY=5e-4
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    return EPOCH,BATCH_SIZE,ITERATION,EPSILON,STEP_SIZE,SEED,DEVICE,M,LR,MOMENT,WEIGHT_DECAY


def accuracy(net, loader, device):
    acc = torchmetrics.Accuracy(task='multiclass', num_classes=10).to(device)
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        ypred = net(xb)
        _ = acc(ypred, yb)
    return acc.compute()