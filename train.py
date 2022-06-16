import os
import matplotlib.pyplot as plt
import numpy as np
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
from utils import init_weights, set_requires_grad
from torch.utils.data import DataLoader
from torchvision import transforms
from statistics import mean

from model import Generator, Discriminator
from data_loader import Dataset, RandomCrop, Resize, Normalization

parser = argparse.ArgumentParser(description='Train Pix2Pix')
parser.add_argument('--crop_size', default=256, type=int, help='training images crop size')
parser.add_argument('--num_epochs', default=100, type=int, help='training epoch number')
parser.add_argument('--scale_factor', default=2, type=int, help='Pix2Pix scale factor')
parser.add_argument('--opts', nargs='+', default=['direction'])
parser.add_argument('--batch_size', default=64, type=int, help='map data batch size')
parser.add_argument('--data_dir', default='./edges2shoes/train/', type=str, help='data dir')

opt = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"]="1,2,3"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('Device:', device)
print('Current cuda device:', torch.cuda.current_device())
print('Count of using GPUs:', torch.cuda.device_count())

# hyper parameters
crop_size = opt.crop_size
num_epochs = opt.num_epochs

transform_train = transforms.Compose([
    Resize(shape=(286, 286, 3)),
    RandomCrop((256, 256)),
    Normalization(mean=0.5, std=0.5)
])

# data loading
train_set = Dataset(opt.data_dir, transform=transform_train)
train_loader = DataLoader(dataset=train_set, num_workers=0, batch_size=opt.batch_size, shuffle=True, drop_last=True)

num_batch_train = int((train_set.__len__() / opt.batch_size) + ((train_set.__len__() / opt.batch_size) != 0))

out_path = 'training_results/SRF_2/'
if not os.path.exists(out_path):
    os.makedirs(out_path)

netG = Generator().to(device)
netD = Discriminator(opt.scale_factor).to(device)

netG = nn.DataParallel(netG).to(device)
netD = nn.DataParallel(netD).to(device)

init_weights(netG)
init_weights(netD)

fn_loss = nn.BCELoss().to(device) # binary cross entropy
l1_loss = nn.L1Loss().to(device)

optimizerG = optim.Adam(netG.parameters(), lr=0.0003, betas=(0.5, 0.999))
optimizerD = optim.Adam(netD.parameters(), lr=0.0003, betas=(0.5, 0.999))

fn_tonumpy = lambda x: x.to('cpu').detach().numpy().transpose(0, 2, 3, 1)
fn_denorm = lambda x, mean, std: (x * std) + mean

dir_log_train = './log'
if not os.path.exists(dir_log_train):
    os.makedirs(dir_log_train)

writer_train = SummaryWriter(log_dir=dir_log_train)

img = plt.imread('./input/input.jpg')[:, :, :3]
img = {'label': img[:, 256:, :], 'input': img[:, :256, :]}

writer_train.add_image('input', img['input'], 1, dataformats='NHWC')
writer_train.add_image('label', img['label'], 1, dataformats='NHWC')

for epoch in range(1, num_epochs + 1):
    netG.train()
    netD.train()

    loss_G_L1_train = []
    loss_G_gan_train = []
    loss_D_fake_train = []
    loss_D_real_train = []

    for batch, data in enumerate(train_loader, 1):
        label = data['label'].to(device)
        input = data['input'].to(device)

        output = netG(input)

        set_requires_grad(netD, True)
        optimizerD.zero_grad()

        real = torch.cat((input, label), 1)
        fake = torch.cat((input, output), 1)

        pred_real = netD(real)
        pred_fake = netD(fake.detach())

        loss_D_real = fn_loss(pred_real, torch.ones_like(pred_real))
        loss_D_fake = fn_loss(pred_fake, torch.zeros_like(pred_fake))

        loss_D = (loss_D_real + loss_D_fake) * 0.5

        loss_D.backward()
        optimizerD.step()

        set_requires_grad(netD, False)
        optimizerG.zero_grad()

        fake = torch.cat((input, output), 1)
        pred_fake = netD(fake)
        loss_g_gan = fn_loss(pred_fake, torch.ones_like(pred_fake))
        loss_g_l1 = l1_loss(output, label)
        loss_g = loss_g_gan + 100 * loss_g_l1

        loss_g.backward()

        optimizerG.step()

        loss_G_L1_train += [loss_g_l1.item()]
        loss_G_gan_train += [loss_g_gan.item()]
        loss_D_real_train += [loss_D_real.item()]
        loss_D_fake_train += [loss_D_fake.item()]

        print("train: epoch %04d / %04d | batch: %04d / %04d | GAN gan %.4f | GAN L1 %.4f | DISC REAL: %.4f | DISC REAL: %.4f"
              % (num_epochs, epoch, num_batch_train, batch, np.mean(loss_G_gan_train), np.mean(loss_G_L1_train), np.mean(loss_D_real_train), np.mean(loss_D_fake_train)))
        
        if batch % 1000 == 0:
            with torch.no_grad():
                netG.eval()
                output = netG(img.to(device))

                output = fn_tonumpy(fn_denorm(output, mean=0.5, std=0.5)).squeeze()
                output = np.clip(output, a_min=0, a_max=1)

                writer_train.add_image('output', output, (int(batch/1000) + (epoch*10)), dataformats='NHWC')
            netG.train()

    writer_train.add_scalar('loss_G_l1', mean(loss_G_L1_train), epoch)
    writer_train.add_scalar('loss_G_gan', mean(loss_G_gan_train), epoch)
    writer_train.add_scalar('loss_D_fake', mean(loss_D_fake_train), epoch)
    writer_train.add_scalar('loss_D_real', mean(loss_D_real_train), epoch)

    netG.eval()
    if epoch % 10 == 0:
        torch.save(netG.state_dict(), 'epochs/netG_epoch_%d_%d.pth' % (4, epoch))


# start 5/3 화 10:01






