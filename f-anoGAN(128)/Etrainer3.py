import os

import torch
from torch import autograd, optim
from torchvision.utils import save_image
import numpy as np

from utils import AverageMeter
import wgan64x64
from sklearn import metrics

class Trainer(object):

    def __init__(self, net_g, net_d, net_e, optimizer_e, dataloader, testdataloader, device, c=10.0):
        self.net_g = net_g
        self.net_d = net_d
        self.net_e = net_e
        self.optimizer_e = optimizer_e
        self.dataloader = dataloader
        self.testdataloader = testdataloader
        self.device = device
        self.c = c
        self.loss_fn = torch.nn.MSELoss(reduction='mean')
        self.test_loss_fn = torch.nn.MSELoss(reduction='none')

    def train(self):
        self.net_e.train()
        self.net_g.eval()
        self.net_d.eval()
        loss_e_meter = AverageMeter()

        for real, _ in self.dataloader:
            # train Encoder

            real = real.to(self.device)
            z = self.net_e(real)

            fake = self.net_g(z)
            recon_features = self.net_d.extract_feature(fake)
            image_features = self.net_d.extract_feature(real)

            loss_e = self.loss_fn(real, fake) + 1*self.loss_fn(recon_features, image_features)
            
            self.optimizer_e.zero_grad()
            loss_e.backward()
            self.optimizer_e.step()

            loss_e_meter.update(loss_e.item(), number=real.size(0))

        return loss_e_meter.average

    def test(self):
        self.net_e.eval()
        self.net_g.eval()
        self.net_d.eval()

        with torch.no_grad():
            y_true, y_score1, y_score2, y_score3 = [], [], [], []
            for (x, label) in self.testdataloader:
                x_data = x.to(self.device)
                score1 = torch.zeros_like(label, dtype=torch.float).to(self.device)
                score2 = torch.zeros_like(label, dtype=torch.float).to(self.device)
                score3 = torch.zeros_like(label, dtype=torch.float).to(self.device)

                z_emb = self.net_e(x_data)
                recon_img = self.net_g(z_emb)
                recon_features = self.net_d.extract_feature(recon_img)
                image_features = self.net_d.extract_feature(x_data)
                z_img_emb = self.net_e(recon_img)

                ##- DISTANCE BASED ON z OF E(Q) AND z OF E(G(E(Q)))
                img_distance = self.test_loss_fn(x_data, recon_img)
                img_distance = torch.flatten(img_distance, start_dim=1)
                img_distance = torch.mean(img_distance, 1)
                loss_fts = self.test_loss_fn(recon_features, image_features)
                loss_fts = torch.flatten(loss_fts, start_dim=1)
                loss_fts = torch.mean(loss_fts, 1)
                # print(img_distance.shape)
                # print(loss_fts.shape)
                dloss = -self.net_d(x_data) + self.net_d(recon_img)
                z_distance = loss_fts
                img_distance2 = img_distance + 1*loss_fts

                y_true.append(label)
                y_score1.append(img_distance.cpu())
                y_score2.append(loss_fts.cpu())
                y_score3.append(img_distance2.cpu())

            y_true = np.concatenate(y_true)
            y_score1 = np.concatenate(y_score1)
            y_score2 = np.concatenate(y_score2)
            y_score3 = np.concatenate(y_score3)

            auc1 = metrics.roc_auc_score(y_true, y_score1)
            auc2 = metrics.roc_auc_score(y_true, y_score2)
            auc3 = metrics.roc_auc_score(y_true, y_score3)

            print('izi: {}'.format(auc1))
            print('izif: {}'.format(auc2))
            print('sum: {}'.format(auc3))
            return auc1, auc2, auc3

    def save_sample(self, filename):
        self.net_e.eval()

        with torch.no_grad():
            for real, _ in self.dataloader:
                real = real.to(self.device)
                z = self.net_e(real)
                fake = self.net_g(z)
                break

        os.makedirs(os.path.dirname(filename), exist_ok=True)
        save_image(fake.data, filename, normalize=True)

    def save_reconstruct(self, filename):
        self.net_e.eval()
        index = 0
        with torch.no_grad():
            for real, i in self.dataloader:
                index+=1
                real = real.to(self.device)
                z = self.net_e(real)
                fake = self.net_g(z)

                os.makedirs(os.path.dirname(filename+str(index)+'.jpg'), exist_ok=True)
                save_image(fake.data, filename+str(index)+'.jpg', normalize=True)
                save_image(real.data, 'reconstruction/original'+str(index)+'.jpg', normalize=True)