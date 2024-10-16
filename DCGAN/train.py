import argparse

import os
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import *
from dataset import *
from util import *

import matplotlib.pyplot as plt

from torchvision import transforms, datasets


## parser 생성
parser = argparse.ArgumentParser(description="Train the DCGAN",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--lr', default=2e-4, type=float, dest='lr')
parser.add_argument('--batch_size', default=128, type=int, dest="batch_size")
parser.add_argument('--num_epoch', default=100, type=int, dest='num_epoch')

parser.add_argument('--data_dir', default='/content/drive/MyDrive/DCGAN/BSDS500/BSDS500/data/images', type=str, dest='data_dir')
parser.add_argument('--ckpt_dir', default='/content/drive/MyDrive/DCGAN/checkpoint', type=str, dest='ckpt_dir')
parser.add_argument('--log_dir', default='/content/drive/MyDrive/DCGAN/log', type=str, dest='log_dir')
parser.add_argument('--result_dir', default='/content/drive/MyDrive/DCGAN/result', type=str, dest='result_dir')

parser.add_argument('--mode', default='train', type=str, dest='mode')
parser.add_argument('--train_continue', default='off', type=str, dest='train_continue')

parser.add_argument('--task', default='DCGAN', choices=['inpainting', 'denoising', 'super_resolution', 'DCGAN'], type=str, dest='task') # []로 묶어주면 무조건 그 중에서 하나 골라야 된다고 함.
parser.add_argument('--opts', nargs='+', default=['bilinear', 4, 0], dest='opts') # nargs='+'로 여러개의 opts를 받을 수 있음.

parser.add_argument('--ny', default=64, type=int, dest='ny')
parser.add_argument('--nx', default=64, type=int, dest='nx')
parser.add_argument('--nch', default=3, type=int, dest='nch')
parser.add_argument('--nker', default=128, type=int, dest='nker')

parser.add_argument('--network', default='DCGAN', choices=['unet', 'hourglass', 'resnet', 'DCGAN'], type=str, dest='network')
parser.add_argument('--learning_type', default='plain', choices=['plain', 'residual'], type=str, dest='learning_type')

args = parser.parse_args()

## parameter 설정
lr = args.lr
batch_size = args.batch_size
num_epoch = args.num_epoch

data_dir = args.data_dir
ckpt_dir = args.ckpt_dir
log_dir = args.log_dir
result_dir = args.result_dir

task = args.task
opts = [args.opts[0], np.asarray(args.opts[1:]).astype(np.float32)]

ny = args.ny
nx = args.nx
nch = args.nch
nker = args.nker

network = args.network
learning_type = args.learning_type

mode = args.mode
train_continue = args.train_continue

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("mode: %s" % mode)

print("learning rate: %.4e" % lr)
print("batch size: %d" % batch_size)
print("number of epoch: %d" % num_epoch)

print("task: %s" % task)
print("opts: %s" % opts)

print("network: %s" % network)
print("learning type: %s" % learning_type)

print("data dir: %s" % data_dir)
print("ckpt dir: %s" % ckpt_dir)
print("log dir: %s" % log_dir)
print("result dir: %s" % result_dir)

print("device: %s" % device)

## 디렉토리 생성
result_dir_train = os.path.join(result_dir, 'train')
result_dir_test = os.path.join(result_dir, 'test')

if not os.path.exists(result_dir):
    os.makedirs(os.path.join(result_dir_train, 'png'))
    os.makedirs(os.path.join(result_dir_test, 'png'))
    os.makedirs(os.path.join(result_dir_test, 'numpy'))

## 네트워크 학습
if mode == 'train':
    transform_train = transforms.Compose([Resize(shape=(ny, nx, nch)), Normalization(mean=0.5, std=0.5)])

    dataset_train = Dataset(data_dir=os.path.join(data_dir,'train'), transform=transform_train, task=task, opts=opts)
    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=2)

    ## 부수적인 변수 설정
    num_data_train = len(dataset_train)

    num_batch_train = int(np.ceil(num_data_train / batch_size))
else:
    transform = transforms.Compose([Resize(shape=(ny,nx,nch)), Normalization(mean=0.5, std=0.5)])

    dataset_test = Dataset(data_dir=os.path.join(data_dir,'test'), transform=transform)
    loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=2)

    num_data_test = len(dataset_test)
    num_batch_test = int(np.ceil(num_data_test / batch_size))

## 네트워크 생성
if network == 'DCGAN':
    netG = DCGAN(in_channels=100, out_channels=nch, nker=nker).to(device)
    netD = Discriminator(in_channels=nch, out_channels=1, nker=nker).to(device)

    init_weights(netG, init_type='normal', init_gain=0.02)
    init_weights(netD, init_type='normal', init_gain=0.02)

## 손실함수 정의
# fn_loss = nn.BCEWithLogitsLoss().to(device) # for segmentation
# fn_loss = nn.MSELoss().to(device) # for regression & restoration
fn_loss = nn.BCELoss().to(device)

## optimizer 설정
optimG = torch.optim.Adam(netG.parameters(), lr=lr, betas=(0.5, 0.999))
optimD = torch.optim.Adam(netD.parameters(), lr=lr, betas=(0.5, 0.999))

## 부수적인 함수 설정
fn_tonumpy = lambda x: x.to('cpu').detach().numpy().transpose(0, 2, 3, 1)
fn_denorm = lambda x, mean, std: (x * std) + mean
# fn_class = lambda x: 1.0 * (x > 0.5)

cmap = None

## Tensorboard을 위한 SummaryWriter 설정
writer_train = SummaryWriter(log_dir=os.path.join(log_dir,'train'))

## 네트워크 학습
st_epoch = 0

## Train mode
if mode == 'train':
    if train_continue == 'on':
        netG, netD, optimG, optimD, st_epoch = load(ckpt_dir=ckpt_dir, netG=netG, netD=netD, optimG=optimG, optimD=optimD)

    for epoch in range(st_epoch + 1, num_epoch + 1):
        netG.train()
        netD.train()
        
        loss_G_train = []
        loss_D_real_train = []
        loss_D_fake_train = []

        for batch, data in enumerate(loader_train, 1):
            # forward pass
            label = data['label'].to(device)
            input = torch.randn(label.shape[0], 100, 1, 1).to(device)

            output = netG(input)

            # backward netD
            set_requires_grad(netD, True)
            optimD.zero_grad()

            pred_real = netD(label)
            pred_fake = netD(output.detach()) # .detach()를 함으로써 back-propagation이 generator까지 가지 못하게 한다고 함. 즉, discriminator만 적용

            loss_D_real = fn_loss(pred_real, torch.ones_like(pred_real))  
            loss_D_fake = fn_loss(pred_fake, torch.zeros_like(pred_fake))
            loss_D = 0.5 * (loss_D_real + loss_D_fake) 
            
            loss_D.backward()
            optimD.step()

            # backward netG
            set_requires_grad(netD, False)
            optimG.zero_grad()

            pred_fake = netD(output)

            loss_G = fn_loss(pred_fake, torch.ones_like(pred_fake))

            loss_G.backward()
            optimG.step()

            # 손실함수 계산
            loss_G_train += [loss_G.item()]
            loss_D_real_train += [loss_D_real.item()]
            loss_D_fake_train += [loss_D_fake.item()]

            print(f"Train: Epoch {epoch:04d} / {num_epoch:04d} | Batch {batch:04d} / {num_batch_train:04d} \
                  | Gen {np.mean(loss_G_train):.4f} | Disc Real: {np.mean(loss_D_real_train)} | Disc Fake: {np.mean(loss_D_fake_train)}")

            # if batch % 20 == 0:
            # Tensorboard 저장하기
            output = fn_tonumpy(fn_denorm(output, mean=0.5, std=0.5)).squeeze()
            output = np.clip(output, a_min=0, a_max=1)
        
            id = num_batch_train * (epoch - 1) + batch

            plt.imsave(os.path.join(result_dir_train, 'png', '%04d_output.png' % id), output[0].squeeze(), cmap=cmap)
            writer_train.add_image('output', output, id, dataformats='NHWC')

        writer_train.add_scalar('loss_G', np.mean(loss_G_train), epoch)
        writer_train.add_scalar('loss_D_real', np.mean(loss_D_real_train), epoch)
        writer_train.add_scalar('loss_D_fake', np.mean(loss_D_fake_train), epoch)

        if epoch % 50 == 0 or epoch == num_epoch:
            save(ckpt_dir=ckpt_dir, netG=netG, netD=netD, optimG=optimG, optimD=optimD, epoch=epoch)

    writer_train.close()

# Test mode
else:
    netG, netD, optimG, optimD, st_epoch = load(ckpt_dir=ckpt_dir, netG=netG, netD=netD, optimG=optimG, optimD=optimD)

    with torch.no_grad():
        netG.eval()

        input = torch.randn(batch_size, 100, 1, 1).to(device)
        output = netG(input)

        output = fn_tonumpy(fn_denorm(output, mean=0.5, std=0.5))

        for j in range(output.shape[0]):
            id = j

            output_ = output[j]
            np.save(os.path.join(result_dir_test, 'numpy', '%04d_output.npy' % id), output_)

            output_ = np.clip(output_, a_min=0, a_max=1)
            plt.imsave(os.path.join(result_dir_test, 'png', '%04d_output.png' % id), output_, cmap=cmap)

