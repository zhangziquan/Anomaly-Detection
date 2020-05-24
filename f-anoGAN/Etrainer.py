import os

import torch
from torch import autograd, optim
from torchvision.utils import save_image

from models import Discriminator, Generator
from utils import AverageMeter
import wgan64x64


class Trainer(object):

    def __init__(self, net_g, net_d, net_e, optimizer_e, dataloader, device, c=10.0):
        self.net_g = net_g
        self.net_d = net_d
        self.net_e = net_e
        self.optimizer_e = optimizer_e
        self.dataloader = dataloader
        self.device = device
        self.c = c
        self.loss_fn = torch.nn.MSELoss(reduction='mean')

    def train(self):
        self.net_e.train()

        loss_e_meter = AverageMeter()

        for real, _ in self.dataloader:
            # train Encoder

            real = real.to(self.device)
            z = self.net_e(real)

            fake = self.net_g(z)
            loss_e = self.loss_fn(real, fake)
                
            self.optimizer_e.zero_grad()
            loss_e.backward()
            self.optimizer_e.step()

            loss_e_meter.update(loss_e.item(), number=real.size(0))

        return loss_e_meter.average

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
                save_image(real.data, 'reconstruction2/original'+str(index)+'.jpg', normalize=True)