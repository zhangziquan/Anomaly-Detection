import argparse

import os
import torch
import csv
from torch import optim
from torch.utils import data
from torchvision import datasets, transforms
import torchvision
import torch
import seaborn as sns
import numpy as np

from models import Discriminator, Generator
from Etrainer import Trainer
from utils import PlotHelper
from wgan64x64 import GoodGenerator, GoodDiscriminator, Encoder
from torchvision.utils import save_image
import matplotlib.pyplot as plt

from torchvision.transforms import ToPILImage

def get_loader(root, batch_size, workers, img_size=64, drop=True):

    trans = torchvision.transforms.Compose([
            torchvision.transforms.Resize(size = img_size),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])    
    dataset = torchvision.datasets.ImageFolder(root, transform=trans)
    
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             num_workers= workers,
                                             drop_last = drop)
    return dataloader

class Anomaly_scoring(object):

    def __init__(self, net_g, net_d, net_e, kappa, dataloader, device, c=10.0):
        self.net_g = net_g
        self.net_d = net_d
        self.net_e = net_e
        self.dataloader = dataloader
        self.device = device
        self.c = c
        self.kappa = kappa
        self.loss_fn = torch.nn.MSELoss(reduction='mean')
        self.heatmap = torch.nn.MSELoss(reduction='none')
        self.filename = './result/result.csv'
        if os.path.isfile(self.filename):
            os.remove(self.filename)

    def ano_score(self):
        self.net_g.eval()
        self.net_d.eval()
        self.net_e.eval()
        with torch.no_grad():
            for i, loader in enumerate(self.dataloader, 0):
                # print(loader)
                x_data, y_data = self.unpack_loader(loader)
                # print(x_data.shape, y_data.shape)
                x_data = x_data.to(self.device)

                z_emb = self.net_e(x_data)
                recon_img = self.net_g(z_emb)
                recon_features = self.net_d.extract_feature(recon_img)
                image_features = self.net_d.extract_feature(x_data)
                z_img_emb = self.net_e(recon_img)

                ##- DISTANCE BASED ON z OF E(Q) AND z OF E(G(E(Q)))
                img_distance = self.loss_fn(x_data, recon_img)
                '''
                heat_map = self.heatmap(x_data, recon_img)
                hp_max = heat_map.max()
                hp_min = heat_map.min()
               # save_image(heat_map, './heatmap/'+str(i)+'.png', normalize=True)
                heat_map = ((heat_map - hp_min) / (hp_max - hp_min + 0.000001))
                #save_image(heat_map, './heatmap/'+str(i)+'1.png', normalize=True)
                np.random.seed(0)
                print(heat_map.shape)
                x = heat_map[0].cpu().numpy()[0] + heat_map[0].cpu().numpy()[1] + heat_map[0].cpu().numpy()[2]
                f, (ax1, ax2) = plt.subplots(figsize=(32,32),nrows=2)
                sns.heatmap(x, annot=False, ax=ax1)
                sns.heatmap(x, annot=True, fmt='.1f', ax=ax2)
                f.savefig('./heatmap/'+str(i)+'.jpg')
                '''
                loss_fts = self.loss_fn(recon_features, image_features)
                z_distance = loss_fts
                img_distance2 = img_distance + self.kappa*loss_fts
                # print(img_distance, z_distance, img_distance2)
                self.save_result(self.filename, y_data, img_distance, z_distance, img_distance2)
        print("Done!")


    def unpack_loader(self, loader):
        data = loader[0].to(self.device)
        label = loader[1].to(self.device)
        return data, label

    def save_result(self, filename, encoding, img_distance, z_distance, img_distance2):
        with open( filename, "a" ) as f:
            writer = csv.writer(f, delimiter=',')
            label = encoding.data.cpu().numpy()
            # print(label[0])
            # print(img_distance.data.cpu().numpy())
            # print(z_distance.data.cpu().numpy())
            writer.writerow([label[0], img_distance.data.cpu().numpy(), z_distance.data.cpu().numpy(), img_distance2.data.cpu().numpy()])


def main():
    torch.random.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='/data/weishizheng/x-ray/rsna-data/test')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--kappa', type=float, default=1)
    config = parser.parse_args()
    device = torch.device('cuda:1')

    # networks
    net_g = GoodGenerator().to(device)
    net_d = GoodDiscriminator().to(device)
    net_e = Encoder(64, 64).to(device)

    # Load models
    name = 'xray_model_07'
    net_g.load_state_dict(torch.load('./models/3rd/gen_' + name + '.pt'))
    net_d.load_state_dict(torch.load('./models/3rd/dis_' + name + '.pt'))
    net_e.load_state_dict(torch.load('./models/3rd/enc2_' + name + '.pt'))

    # print(net_g)
    # print(net_d)
    # print(net_e)

    # data loader
    dataloader = get_loader(config.root, config.batch_size, config.workers)
    ano_sco = Anomaly_scoring(net_g, net_d, net_e, config.kappa, dataloader, device)
    ano_sco.ano_score()



if __name__ == '__main__':
    main()
