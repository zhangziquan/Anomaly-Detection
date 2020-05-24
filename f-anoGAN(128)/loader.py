from torch.utils import data
from torchvision import datasets, transforms
import torchvision
import torch


def get_loader(root, batch_size, workers, img_size=64, drop=True, crop = False):
    if (crop == True):
        trans = torchvision.transforms.Compose([
            torchvision.transforms.CenterCrop((800,800)),
            torchvision.transforms.Resize(size = img_size),# 缩放图片(Image)，保持长宽比不变，最短边为img_size像素
            #torchvision.transforms.Grayscale(num_output_channels=1),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
    else:
        trans = torchvision.transforms.Compose([
            # torchvision.transforms.CenterCrop((784,784)),
            torchvision.transforms.Resize(size = img_size),# 缩放图片(Image)，保持长宽比不变，最短边为img_size像素
            #torchvision.transforms.Grayscale(num_output_channels=1),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

    dataset = torchvision.datasets.ImageFolder(root, transform=trans)
    
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             num_workers= workers,
                                             drop_last = drop) #drap_last = True:如果样本总量除于batch_size后有余数,则丢弃余数部分
    return dataloader