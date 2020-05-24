from torch.utils import data
from torchvision import datasets, transforms
import torchvision
import torch


def get_loader(root, batch_size, workers, img_size=64, drop=True, crop = False):
    if (crop == True):
        trans = torchvision.transforms.Compose([
            torchvision.transforms.CenterCrop((800,800)),
            torchvision.transforms.Resize(size = img_size),# ����ͼƬ(Image)�����ֳ���Ȳ��䣬��̱�Ϊimg_size����
            #torchvision.transforms.Grayscale(num_output_channels=1),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
    else:
        trans = torchvision.transforms.Compose([
            # torchvision.transforms.CenterCrop((784,784)),
            torchvision.transforms.Resize(size = img_size),# ����ͼƬ(Image)�����ֳ���Ȳ��䣬��̱�Ϊimg_size����
            #torchvision.transforms.Grayscale(num_output_channels=1),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

    dataset = torchvision.datasets.ImageFolder(root, transform=trans)
    
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             num_workers= workers,
                                             drop_last = drop) #drap_last = True:���������������batch_size��������,������������
    return dataloader