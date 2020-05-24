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
import cv2
from torchvision.transforms import ToPILImage
from PIL import Image
import math
from torch.autograd import Variable


class MyGaussianBlur():

    def __init__(self, radius=1, sigema=1.5):
        self.radius=radius
        self.sigema=sigema   

    def calc(self,x,y):
        res1=1/(2*math.pi*self.sigema*self.sigema)
        res2=math.exp(-(x*x+y*y)/(2*self.sigema*self.sigema))
        return res1*res2

    def template(self):
        sideLength=self.radius*2+1
        result = np.zeros((sideLength, sideLength))
        for i in range(sideLength):
            for j in range(sideLength):
                result[i,j]=self.calc(i-self.radius, j-self.radius)
        all=result.sum()
        return result/all    

    def filter(self, image, template): 
        arr=np.array(image)
        height=arr.shape[0]
        width=arr.shape[1]
        chanel = arr.shape[2]
        newData=np.zeros((height, width, chanel))
        for z in range(chanel):
            for i in range(self.radius, height-self.radius):
                for j in range(self.radius, width-self.radius):
                    t=arr[i-self.radius:i+self.radius+1, j-self.radius:j+self.radius+1, z]
                    a= np.multiply(t, template)
                    newData[i, j, z] = a.sum()          
        return newData
 
def random_rotate(imgs):
    labels = []
    rets = torch.zeros_like(imgs)
    for i in range(imgs.size(0)):
        num = random.randint(0,3)
        labels.append(num)
        if(num== 0):
            rets[i] = imgs[i]
        else:
            num = num*2+1
            rets[i] = rotate_img(imgs[i], num)
    return rets, torch.LongTensor(labels)


def detm_rotate(imgs, deg):
    labels = []
    rets = torch.zeros_like(imgs)
    for i in range(imgs.size(0)):
        labels.append(deg)
        if(deg== 0):
            rets[i] = imgs[i]
        else:
            num= deg*2+1
            rets[i] = rotate_img(imgs[i], num)
    return rets, torch.LongTensor(labels)


def rotate_img(img_tensor, deg):
    # save_image(img_tensor.data, './original.png', normalize=True)
    img = np.asarray(img_tensor.permute(1, 2, 0))
    # print(img.shape)
    GBlur=MyGaussianBlur(radius=3, sigema=deg)
    temp=GBlur.template()
    img=GBlur.filter(img, temp)
    # print(img.shape)
    im = torch.tensor(img).permute(2, 0, 1)
    # save_image(im.data, './sample1.png', normalize=True)
    '''
    if(deg == 1):
        save_image(im.data, './sample1.png', normalize=True)
    if(deg == 2):
        save_image(im.data, './sample3.png', normalize=True)
    if(deg == 3):
        save_image(im.data, './sample5.png', normalize=True)
    if(deg == 4):
        save_image(im.data, './sample7.png', normalize=True)
    '''

    return im


def my_dataloader(fpath,img_size = 32,batch_size = 64, workers = 4, drop = False, crop = False):
    """    args:
        fpath: str
        img_size: seq or int
        batch_size: int
        workers: int
    return:
        torch.utils.data.dataloader.DataLoader;
    """
    if (crop == True):
        trans = torchvision.transforms.Compose([
            torchvision.transforms.CenterCrop((800,800)),
            torchvision.transforms.Resize(size = img_size),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
    else:
        trans = torchvision.transforms.Compose([
            torchvision.transforms.Resize(size = img_size),
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
    NUM_CLASSES = 4
    model = baseline.resnet32(num_classes=NUM_CLASSES).to(device)
    # model.load_state_dict(torch.load('cropmodels/{}.pth'.format(options.m)))
    train_loader = my_dataloader(r"/data/weishizheng/x-ray/rsna-data/train", (64, 64), drop = False, crop = True)
    optimizer = optim.RMSprop(model.parameters(), options.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 50)

    for e in range(options.e):
        losses = []
        model.train()
        acc, num = 0, 0

        for (x, _) in train_loader:
            x, label = random_rotate(x)
            x, label = x.to(device), label.to(device)
            out = model(x)
            out_ = out.softmax(dim=1)
            loss = F.cross_entropy(out, label)
            out_ = np.argmax(out_.detach().cpu().numpy(), axis=1)
            acc += np.equal(out_, label.cpu()).numpy().sum()
            num += len(label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        scheduler.step()

        print('acc: {:.2f}%, {}/{}'.format(acc/num*100, int(acc), num))
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
    torch.save(model.state_dict(), 'cropmodels/{}.pth'.format(options.m))


def test():
    NUM_CLASSES = 4
    model = baseline.resnet32(num_classes=NUM_CLASSES).to(device)
    model.load_state_dict(torch.load('cropmodels/{}.pth'.format(options.m)))
    model.eval()
    val_loader = my_dataloader(r"/data/weishizheng/x-ray/rsna-data/test02", (64, 64), drop = False, crop = True)

    with torch.no_grad():
        y_true, y_score = [], []
        for (x, label) in val_loader:
            score = torch.zeros_like(label, dtype=torch.float).to(device)
            # c = options.t
            for c in range(4):
                x_, labele = detm_rotate(x, c)
                x_ = x_.to(device)
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
    parser.add_argument('--model', dest='m', type=int, default=0)
    parser.add_argument('--epoch', dest='e', type=int, default=0)
    parser.add_argument('--lr', type=float, default=1e-1)
    options = parser.parse_args()
    device = torch.device('cuda:3')
    if not options.eval:
        train()
    else:
        test()
