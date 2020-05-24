import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import random
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
import cv2
from torchvision.transforms import ToPILImage
from PIL import Image
import math
from torch.autograd import Variable
from net import AutoEncoder
import os
import csv


def my_dataloader(fpath,img_size = 64,batch_size = 64, workers = 4, drop = False):
    """    args:
        fpath: str
        img_size: seq or int
        batch_size: int
        workers: int
    return:
        torch.utils.data.dataloader.DataLoader;
    """
    trans = torchvision.transforms.Compose([
            torchvision.transforms.Resize(size = img_size),
#            tv.transforms.CenterCrop(img_size),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
    
    dataset = torchvision.datasets.ImageFolder(fpath, transform=trans)
    
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             num_workers= workers,
                                             drop_last = drop) #drap_last = True:
    return dataloader


def train():
    model = AutoEncoder(128,64*64*3).to(device)
    # model.load_state_dict(torch.load('rotmodels/{}.pth'.format(options.m)))
    train_loader = my_dataloader(r"/data/weishizheng/x-ray/rsna-data/train", (64, 64),32, drop = False)
    optimizer = optim.Adam(model.parameters(), 1e-4, betas=(0.0, 0.9))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 50)

    for e in range(options.e):
        losses = []
        model.train()

        for (x, label) in train_loader:
            x, label = x.to(device), label.to(device)
            out_e, out_d = model(x)
            loss_fn = torch.nn.MSELoss(reduction='mean')
            loss = loss_fn(x, out_d)
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
    torch.save(model.state_dict(), 'models/{}.pth'.format(options.m))

def save_sample(fake, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    save_image(fake.data, filename, normalize=True)


def test():

    model = AutoEncoder(128,64*64*3).to(device)
    model.load_state_dict(torch.load('models/{}.pth'.format(options.m)))
    model.eval()
    val_loader = my_dataloader(r"/data/weishizheng/x-ray/rsna-data/test03", (64, 64), drop = False)
    epoch = 0
    with torch.no_grad():
        y_true, y_score = [], []
        for (x, label) in val_loader:
            x = x.to(device)
            out_e, out_d = model(x)
            save_sample(out_d,'reconstruction/reconstruction_{:02d}.jpg'.format(epoch + 1))
            save_sample(x,'reconstruction/original_{:02d}.jpg'.format(epoch + 1))

            loss_fn = torch.nn.MSELoss(reduction='none')
            score = loss_fn(x, out_d)
            score = torch.flatten(score, start_dim=1)
            score = torch.mean(score, 1)

            y_true.append(label)
            y_score.append(score.cpu())
            epoch = epoch +1

        y_true = np.concatenate(y_true)
        y_score = np.concatenate(y_score)
        print(len(y_true))
        print(len(y_score))
        plt.hist(y_score[y_true==0], bins=100, density=True, color='green', alpha=0.5)
        plt.hist(y_score[y_true==1], bins=100, density=True, color='red', alpha=0.5)
        save_result('./result/result.csv', y_true, y_score)
        plt.savefig('result.png')
        plt.show()
        auc = metrics.roc_auc_score(y_true, y_score)
        print(auc)

def save_result(filename, encoding, y_score):
    if os.path.isfile(filename):
        os.remove(filename)

    with open( filename, "a" ) as f:
        writer = csv.writer(f, delimiter=',')
        for i in range(len(encoding)):
            writer.writerow([encoding[i], y_score[i]])


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--eval', dest='eval', action='store_true')
    parser.add_argument('--model', dest='m', type=int, default=0)
    parser.add_argument('--epoch', dest='e', type=int, default=40)
    options = parser.parse_args()
    device = torch.device('cuda:3')
    if not options.eval:
        train()
    else:
        test()
