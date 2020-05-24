import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import random
import baseline
from tqdm import tqdm
from argparse import ArgumentParser
import torchvision
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader, sampler
from sklearn import metrics, manifold, decomposition, covariance, mixture, neighbors
from mpl_toolkits.mplot3d import Axes3D
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.determinstic = False
BATCH_SIZE = 64


def random_rotate(imgs):
    labels = []
    rets = torch.zeros_like(imgs)
    for i in range(imgs.size(0)):
        num = random.randint(0, 3)
        labels.append(num)
        rets[i] = rotate_img(imgs[i], num*90)
    return rets, torch.LongTensor(labels)


def detm_rotate(imgs, deg):
    labels = []
    rets = torch.zeros_like(imgs)
    for i in range(imgs.size(0)):
        labels.append(deg//90)
        rets[i] = rotate_img(imgs[i], deg)
    return rets, torch.LongTensor(labels)


def rotate_img(img_tensor, deg):
    if deg == 0:
        return img_tensor
    if deg == 90:
        return img_tensor.rot90(1, (1, 2))
    if deg == 180:
        return img_tensor.rot90(2, (1, 2))
    if deg == 270:
        return img_tensor.rot90(3, (1, 2))

def my_dataloader(fpath,img_size = 32,batch_size = 64, workers = 4, drop = False):
    """    args:
        fpath: str
        img_size: seq or int
        batch_size: int
        workers: int
    return:
        torch.utils.data.dataloader.DataLoader;
    """
    trans = torchvision.transforms.Compose([
            torchvision.transforms.CenterCrop((784,784)),
            torchvision.transforms.Resize(size = img_size),
#            tv.transforms.CenterCrop(img_size),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
    
    dataset = torchvision.datasets.ImageFolder(fpath, transform=trans)
    
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             num_workers= workers,
                                             drop_last = drop) #drap_last = True:
    return dataloader

def one_class_dataloader(c, nw, bs):
    transform1 = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        # transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    train = datasets.CIFAR10('./data', transform=transform1, download=True)
    labels = np.array(train.train_labels)
    class_indices = np.argwhere(labels == c)
    class_indices = class_indices.reshape(class_indices.shape[0])
    trainloader = DataLoader(
        train, bs, sampler=sampler.SubsetRandomSampler(class_indices),
        num_workers=nw, pin_memory=True, drop_last=True)
    transform2 = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    val = datasets.CIFAR10(
        './data', train=False, transform=transform2, download=False)
    valloader = DataLoader(val, 1000, num_workers=nw,
                           pin_memory=True, drop_last=True)

    return trainloader, valloader


def train():
    NUM_CLASSES = 4
    model = baseline.resnet20(num_classes=NUM_CLASSES).to(device)
    train_loader = my_dataloader(r"/data/weishizheng/x-ray/rsna-data/train", (64, 64), drop = False)
    optimizer = optim.SGD(model.parameters(), 1e-1,
                          momentum=0.9, nesterov=True, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 50)

    for e in range(20):
        losses = []
        model.train()
        for (x, _) in train_loader:
            x, label = random_rotate(x)
            x, label = x.to(device), label.to(device)
            out = model(x)
            print(out.shape, label.shape)
            loss = F.cross_entropy(out, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        scheduler.step()

        print(e, np.mean(losses))
        # model.eval()
        # with torch.no_grad():
        #     y_true, y_score = [], []
        #     for (x, label) in val_loader:
        #         x = x.to(device)
        #         score = torch.zeros_like(label, dtype=torch.float).to(device)
        #         for c in range(NUM_CLASSES):
        #             x_ = detm_rotate(x, c * 90)
        #             # x_, _ = detm_mask(x, c)
        #             out = model(x_).softmax(dim=1)
        #             score += out[:, c]
        #         label[label != options.c] = -1
        #         label[label == options.c] = 1
        #         y_true.append(label)
        #         y_score.append(score.cpu())
        #     y_true = np.concatenate(y_true)
        #     y_score = np.concatenate(y_score)
        #     auc = metrics.roc_auc_score(y_true, y_score)
        #     print(auc)
    torch.save(model.state_dict(), 'rotmodels/{}.pth'.format(options.c))


def test():
    NUM_CLASSES = 4
    model = baseline.resnet20(num_classes=NUM_CLASSES).to(device)
    model.load_state_dict(torch.load('rotmodels/{}.pth'.format(options.c)))
    model.eval()
    val_loader = my_dataloader(r"/data/weishizheng/x-ray/rsna-data/test02", (64, 64), drop = False)

    with torch.no_grad():
        y_true, y_score = [], []
        for (x, label) in val_loader:
            x = x.to(device)
            score = torch.zeros_like(label, dtype=torch.float).to(device)
            c = options.t
            for c in range(4):
                x_, labele = detm_rotate(x, c * 90)
                labele = labele.to(device)
                # x_, _ = detm_mask(x, c)
                out  = model(x_)
                out = out.softmax(dim=1)
                loss = F.cross_entropy(out, labele, reduction='none')
                score += loss
            label[label != options.c] = 1
            label[label == options.c] = 0
            y_true.append(label)
            y_score.append(score.cpu())
        y_true = np.concatenate(y_true)
        y_score = np.concatenate(y_score)
        print(len(y_true))
        print(len(y_score))
        plt.hist(y_score[y_true==0], bins=100, density=True, color='green', alpha=0.5)
        plt.hist(y_score[y_true==1], bins=100, density=True, color='red', alpha=0.5)
        plt.savefig('result.png')
        plt.show()
        auc = metrics.roc_auc_score(y_true, y_score)
        print(auc)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--class', dest='c', type=int, default=0)
    parser.add_argument('--eval', dest='eval', action='store_true')
    parser.add_argument('--tria', dest='t', type=int, default=0)
    options = parser.parse_args()
    device = torch.device('cuda:2')
    if not options.eval:
        train()
    else:
        test()
