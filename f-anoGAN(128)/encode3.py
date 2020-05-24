import argparse

import torch
from torch import optim

from loader import get_loader
from Etrainer3 import Trainer
from utils import PlotHelper
from wgan64x64 import GoodGenerator, GoodDiscriminator, Encoder


def main():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='/data/weishizheng/x-ray/rsna-data/train')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=20)
    config = parser.parse_args()
    device = torch.device('cuda:2')

    # networks
    net_g = GoodGenerator().to(device)
    net_d = GoodDiscriminator().to(device)
    net_e = Encoder(64, 128).to(device)

    # Load models
    name = 'xray_model_11'
    net_e.load_state_dict(torch.load('./models/enc3_' + name + '.pt'))
    net_g.load_state_dict(torch.load('./models/gen_' + name + '.pt'))
    net_d.load_state_dict(torch.load('./models/dis_' + name + '.pt'))

    # print(net_g)
    # print(net_d)

    # optimizer
    optimizer_e = optim.Adam(net_e.parameters(), lr=config.lr, betas=(0.0, 0.9))

    # print(optimizer_e)

    # data loader
    dataloader = get_loader(config.root, config.batch_size, config.workers)
    testdataloader = get_loader('/data/weishizheng/x-ray/rsna-data/test03', config.batch_size, config.workers)

    trainer = Trainer(net_g, net_d, net_e, optimizer_e, dataloader, testdataloader, device)
    plotter = PlotHelper('samples/loss.html')
    auc1_avg , auc2_avg , auc3_avg  = 0, 0, 0
    for epoch in range(config.epochs):
        auc1 , auc2 , auc3 = 0, 0, 0
        loss_e = trainer.train()
        auc1 , auc2 , auc3 = trainer.test()
        auc1_avg = auc1 + auc1_avg
        auc2_avg = auc2 + auc2_avg
        auc3_avg = auc3 + auc3_avg
        print('Train epoch: {}/{},'.format(epoch + 1, config.epochs),
              'loss e: {:.6f}.'.format(loss_e))

        trainer.save_sample('reconstruction/sample_{:02d}.jpg'.format(epoch + 1))
        torch.save(trainer.net_e.state_dict(), './models/enc3_' + name +str(epoch)+ '.pt')

    print(auc1_avg/40 , auc2_avg/40 , auc3_avg/40 )
    trainer.save_reconstruct('reconstruction/reconstruct_')

    # Save models
    name = 'new'
    torch.save(trainer.net_e.state_dict(), './models/enc3_' + name +str(epoch)+ '.pt')

if __name__ == '__main__':
    main()
