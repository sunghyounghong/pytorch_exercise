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
parser = argparse.ArgumentParser(description="Train the Unet",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--lr', default=1e-3, type=float, dest='lr')
parser.add_argument('--batch_size', default=4, type=int, dest="batch_size")
parser.add_argument('--num_epoch', default=100, type=int, dest='num_epoch')

parser.add_argument('--data_dir', default='/content/drive/MyDrive/u-net/BSDS500/BSDS500/data/images', type=str, dest='data_dir')
parser.add_argument('--ckpt_dir', default='/content/drive/MyDrive/u-net/checkpoint', type=str, dest='ckpt_dir')
parser.add_argument('--log_dir', default='/content/drive/MyDrive/u-net/log', type=str, dest='log_dir')
parser.add_argument('--result_dir', default='/content/drive/MyDrive/u-net/result', type=str, dest='result_dir')

parser.add_argument('--mode', default='train', type=str, dest='mode')
parser.add_argument('--train_continue', default='off', type=str, dest='train_continue')

parser.add_argument('--task', default='super_resolution', choices=['inpainting', 'denoising', 'super_resolution'], type=str, dest='task') # []로 묶어주면 무조건 그 중에서 하나 골라야 된다고 함.
parser.add_argument('--opts', nargs='+', default=['bilinear', 4], dest='opts') # nargs='+'로 여러개의 opts를 받을 수 있음.

parser.add_argument('--ny', default=320, type=int, dest='ny')
parser.add_argument('--nx', default=480, type=int, dest='nx')
parser.add_argument('--nch', default=3, type=int, dest='nch')
parser.add_argument('--nker', default=64, type=int, dest='nker')

parser.add_argument('--network', default='unet', choices=['unet', 'hourglass'], type=str, dest='network')
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
result_dir_val = os.path.join(result_dir, 'val')
result_dir_test = os.path.join(result_dir, 'test')

if not os.path.exists(result_dir):
    os.makedirs(os.path.join(result_dir_train, 'png'))
    os.makedirs(os.path.join(result_dir_val, 'png'))
    os.makedirs(os.path.join(result_dir_test, 'png'))
    os.makedirs(os.path.join(result_dir_test, 'numpy'))

## 네트워크 학습
if mode == 'train':
    transform_train = transforms.Compose([RandomCrop(shape=(ny, nx)), Normalization(mean=0.5, std=0.5), RandomFlip()])
    transform_val = transforms.Compose([RandomCrop(shape=(ny, nx)), Normalization(mean=0.5, std=0.5)])

    dataset_train = Dataset(data_dir=os.path.join(data_dir, 'train'), transform=transform_train, task=task, opts=opts)
    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=2)

    dataset_val = Dataset(data_dir=os.path.join(data_dir, 'val'), transform=transform_val, task=task, opts=opts)
    loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=2)

    ## 부수적인 변수 설정
    num_data_train = len(dataset_train)
    num_data_val = len(dataset_val)

    num_batch_train = int(np.ceil(num_data_train / batch_size))
    num_batch_val = int(np.ceil(num_data_val / batch_size))
else:
    transform = transforms.Compose([RandomCrop(shape=(ny,nx)), Normalization(mean=0.5, std=0.5)])

    dataset_test = Dataset(data_dir=os.path.join(data_dir,'test'), transform=transform)
    loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=2)

    num_data_test = len(dataset_test)
    num_batch_test = int(np.ceil(num_data_test / batch_size))

## 네트워크 생성
if network == "unet":
    net = UNet(nch=nch, nker=nker, learning_type=learning_type).to(device)
elif network == 'hourglass':
    net = Hourglass(nch=nch, nker=nker, learning_type=learning_type).to(device)

## 손실함수 정의
# fn_loss = nn.BCEWithLogitsLoss().to(device) # for segmentation
fn_loss = nn.MSELoss().to(device) # for regression & restoration

## optimizer 설정
optim = torch.optim.Adam(net.parameters(), lr=lr)

## 부수적인 함수 설정
fn_tonumpy = lambda x: x.to('cpu').detach().numpy().transpose(0, 2, 3, 1)
fn_denorm = lambda x, mean, std: (x * std) + mean
# fn_class = lambda x: 1.0 * (x > 0.5)

cmap = None

## Tensorboard을 위한 SummaryWriter 설정
writer_train = SummaryWriter(log_dir=os.path.join(log_dir,'train'))
writer_val = SummaryWriter(log_dir=os.path.join(log_dir,'val'))

## 네트워크 학습
st_epoch = 0

## Train mode
if mode == 'train':
    if train_continue == 'on':
        net, optim, st_epoch = load(ckpt_dir=ckpt_dir, net=net, optim=optim)

    for epoch in range(st_epoch + 1, num_epoch + 1):
        net.train()
        loss_arr = []

        for batch, data in enumerate(loader_train, 1):
            # forward pass
            label = data['label'].to(device)
            input = data['input'].to(device)

            output = net(input)

            # backward pass
            optim.zero_grad()

            loss = fn_loss(output, label)
            loss.backward()

            optim.step()

            # 손실함수 계산
            loss_arr += [loss.item()]

            print(f"Train: Epoch {epoch:04d} / {num_epoch:04d} | Batch {batch:04d} / {num_batch_train:04d} | Loss {np.mean(loss_arr):.4f}")

            # Tensorboard 저장하기
            label = fn_tonumpy(fn_denorm(label, mean=0.5, std=0.5))
            input = fn_tonumpy(fn_denorm(input, mean=0.5, std=0.5))
            output = fn_tonumpy(fn_denorm(output, mean=0.5, std=0.5))

            input = np.clip(input, a_min=0, a_max=1)
            output = np.clip(output, a_min=0, a_max=1)

            id = num_batch_train * (epoch - 1) + batch

            if id % 10 == 0:

                plt.imsave(os.path.join(result_dir_train, 'png', '%04d_label.png' % id), label[0].squeeze(), cmap=cmap)
                plt.imsave(os.path.join(result_dir_train, 'png', '%04d_input.png' % id), input[0].squeeze(), cmap=cmap)
                plt.imsave(os.path.join(result_dir_train, 'png', '%04d_output.png' % id), output[0].squeeze(), cmap=cmap)

            # writer_train.add_image('label', label, num_batch_train * (epoch - 1) + batch, dataformats='NHWC')
            # writer_train.add_image('input', input, num_batch_train * (epoch - 1) + batch, dataformats='NHWC')
            # writer_train.add_image('output', output, num_batch_train * (epoch - 1) + batch, dataformats='NHWC')

        writer_train.add_scalar('loss', np.mean(loss_arr), epoch)

        with torch.no_grad():
            net.eval()
            loss_arr = []

            for batch, data in enumerate(loader_val, 1):
                # forward pass
                label = data['label'].to(device)
                input = data['input'].to(device)

                output = net(input)

                # loss calculation
                loss = fn_loss(output, label)

                loss_arr += [loss.item()]

                print(f"Valid: Epoch {epoch:04d} / {num_epoch:04d} | Batch {batch:04d} / {num_batch_val:04d} | Loss {np.mean(loss_arr):.4f}")

                # tensorbaord 저장
                label = fn_tonumpy(fn_denorm(label, mean=0.5, std=0.5))
                input = fn_tonumpy(fn_denorm(input, mean=0.5, std=0.5))
                output = fn_tonumpy(fn_denorm(output, mean=0.5, std=0.5))

                input = np.clip(input, a_min=0, a_max=1)
                output = np.clip(output, a_min=0, a_max=1)

                id = num_batch_val * (epoch - 1) + batch

                if id % 5 == 0:

                    plt.imsave(os.path.join(result_dir_val, 'png', f'{id:04d}_label.png'), label[0].squeeze(), cmap=cmap)
                    plt.imsave(os.path.join(result_dir_val, 'png', f'{id:04d}_input.png'), input[0].squeeze(), cmap=cmap)
                    plt.imsave(os.path.join(result_dir_val, 'png', f'{id:04d}_output.png'), output[0].squeeze(), cmap=cmap)

                # writer_val.add_image('label', label, num_batch_val * (epoch - 1) + batch, dataformats='NHWC')
                # writer_val.add_image('input', input, num_batch_val * (epoch - 1) + batch, dataformats='NHWC')
                # writer_val.add_image('output', output, num_batch_val * (epoch - 1) + batch, dataformats='NHWC')

            writer_val.add_scalar('loss', np.mean(loss_arr), epoch)

            if epoch % 50 == 0:
                save(ckpt_dir=ckpt_dir, net=net, optim=optim, epoch=epoch)

        writer_train.close()
        writer_val.close()

# Test mode
else:
    net, optim, st_epoch = load(ckpt_dir=ckpt_dir, net=net, optim=optim)

    with torch.no_grad():
        net.eval()
        loss_arr = []

        for batch, data in enumerate(loader_test, 1):
            # forward pass
            label = data['label'].to(device)
            input = data['input'].to(device)

            output = net(input)

            # loss calculation
            loss = fn_loss(output, label)

            loss_arr += [loss.item()]

            print(f"Test: Batch {batch:04d} / {num_batch_test:04d} | Loss {np.mean(loss_arr):.04f}")

            # Tensorboard 저장
            label = fn_tonumpy(fn_denorm(label, mean=0.5, std=0.5))
            input = fn_tonumpy(fn_denorm(input, mean=0.5, std=0.5))
            output = fn_tonumpy(fn_denorm(output, mean=0.5, std=0.5))

            for j in range(label.shape[0]):
                id = num_batch_test * (batch - 1) + j

                # label_ = label[j]
                # input_ = input[j]
                # output_ = output[j]

                # np.save(os.path.join(result_dir_test, 'numpy', '%04d_label.npy' % id), label_)
                # np.save(os.path.join(result_dir_test, 'numpy', '%04d_input.npy' % id), input_)
                # np.save(os.path.join(result_dir_test, 'numpy', '%04d_output.npy' % id), output_)

                if id % 10 == 0:

                    label_ = np.clip(label_, a_min=0, a_max=1)
                    input_ = np.clip(input_, a_min=0, a_max=1)
                    output_ = np.clip(output_, a_min=0, a_max=1)

                    plt.imsave(os.path.join(result_dir_test, 'png', '%04d_label.png' % id), label_, cmap=cmap)
                    plt.imsave(os.path.join(result_dir_test, 'png', '%04d_input.png' % id), input_, cmap=cmap)
                    plt.imsave(os.path.join(result_dir_test, 'png', '%04d_output.png' % id), output_, cmap=cmap)

    print(f"AVERAGE TEST: BATCH {batch:04d} / {num_batch_test:04d} | LOSS {np.mean(loss_arr):.4f}")
