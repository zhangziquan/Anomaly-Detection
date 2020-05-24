import argparse

import torch
from torch import optim

from loader import get_loader
from models import Discriminator, Generator
from Etrainer2 import Trainer
from utils import PlotHelper
from wgan64x64 import GoodGenerator, GoodDiscriminator, Encoder


def main():
    torch.random.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='/data/weishizheng/x-ray/rsna-data/train')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=20)
    config = parser.parse_args()
    device = torch.device('cuda:2')

    # networks
    net_g = GoodGenerator().to(device)
    net_d = GoodDiscriminator().to(device)
    net_e = Encoder(64, 64).to(device)

    # Load models
    name = 'xray_model_07'
    net_g.load_state_dict(torch.load('./models/3rd/gen_' + name + '.pt'))
    net_d.load_state_dict(torch.load('./models/3rd/dis_' + name + '.pt'))

    print(net_g)
    print(net_d)

    # optimizer
    optimizer_e = optim.Adam(net_e.parameters(), lr=config.lr)

    print(optimizer_e)

    # data loader
    dataloader = get_loader(config.root, config.batch_size, config.workers)

    trainer = Trainer(net_g, net_d, net_e, optimizer_e, dataloader, device)
    plotter = PlotHelper('samples/loss.html')
    for epoch in range(config.epochs):
        loss_e = trainer.train()

        print('Train epoch: {}/{},'.format(epoch + 1, config.epochs),
              'loss e: {:.6f}.'.format(loss_e))

        trainer.save_sample('reconstruction2/sample_{:02d}.jpg'.format(epoch + 1))

    trainer.save_reconstruct('reconstruction2/reconstruct_')

    # Save models
    name = 'xray_model_07'
    torch.save(trainer.net_e.state_dict(), './models/3rd/enc2_' + name + '.pt')


if __name__ == '__main__':
    main()
