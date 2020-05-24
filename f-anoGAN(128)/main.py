import argparse

import torch
from torch import optim

from loader import get_loader
from trainer import Trainer
from utils import PlotHelper
from wgan64x64 import GoodGenerator, GoodDiscriminator


def main():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='/data/weishizheng/x-ray/rsna-data/train')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--dstep', type=int, default=5)
    config = parser.parse_args()
    device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')

    # networks
    net_g = GoodGenerator().to(device)
    net_d = GoodDiscriminator().to(device)

    # Load models
    name = 'xray_model_12'
    net_g.load_state_dict(torch.load('./models/gen_' + name + '.pt'))
    net_d.load_state_dict(torch.load('./models/dis_' + name + '.pt'))

    # optimizer
    optimizer_g = optim.Adam(net_g.parameters(), lr=config.lr, betas=(0.0, 0.9))
    optimizer_d = optim.Adam(net_d.parameters(), lr=config.lr, betas=(0.0, 0.9))

    # data loader
    dataloader = get_loader(config.root, config.batch_size, config.workers)

    trainer = Trainer(net_g, net_d, optimizer_g, optimizer_d, dataloader, device)
    plotter = PlotHelper('samples/loss.html')
    for epoch in range(config.epochs):
        loss_g, loss_d = trainer.train()

        print('Train epoch: {}/{},'.format(epoch + 1, config.epochs),
              'loss g: {:.6f}, loss d: {:.6f}.'.format(loss_g, loss_d))

        trainer.save_sample('samples/sample_{:02d}.jpg'.format(epoch + 1))
        plotter.append(loss_g, loss_d, epoch + 1)

    # Save models
    name = 'xray_model_13'
    torch.save(trainer.net_g.state_dict(), './models/gen_' + name + '.pt')
    torch.save(trainer.net_d.state_dict(), './models/dis_' + name + '.pt')



if __name__ == '__main__':
    main()
