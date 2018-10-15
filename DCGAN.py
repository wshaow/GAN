# -*- coding: utf-8 -*-
"""
    创建人：wshaow
    创建时间：2018年8月13日
    mxnet上的dcgan的代码实例
    学习点：
    1、通过url下载数据集
    2、通过tarfile模块解压文件， 以及基本的文件操作
    3、通过os.walk 遍历某个目录下的所有文件
    4、nd.concatenate(img_list) 将list组合为一个ndarray类型
    5、这里构建网络的方式， 以及如何使用lekyrelu
    6、nn.Conv2DTranspose的用途， 卷积转置， 论坛上有人说现在基本上直接使用双线性插值来代替这个进行上采样
"""
from __future__ import print_function
import os
import matplotlib as mpl
import tarfile
import matplotlib.image as mpimg
from matplotlib import pyplot as plt

import mxnet as mx
from mxnet import gluon
from mxnet import ndarray as nd
from mxnet.gluon import nn, utils
from mxnet import autograd
import numpy as np

from datetime import datetime
import time
import logging


def transform(data, target_wd, target_ht):
    # resize to target_wd * target_ht
    data = mx.image.imresize(data, target_wd, target_ht)
    # transpose from (target_wd, target_ht, 3)
    # to (3, target_wd, target_ht)
    data = nd.transpose(data, (2, 0, 1))
    # normalize to [-1, 1]
    data = data.astype(np.float32)/127.5 - 1
    # if image is greyscale, repeat 3 times to get RGB image.
    if data.shape[0] == 1:
        data = nd.tile(data, (3, 1, 1))
    return data.reshape((1,) + data.shape)


def visualize(img_arr):
    plt.imshow(((img_arr.asnumpy().transpose(1, 2, 0) + 1.0) * 127.5).astype(np.uint8))
    plt.axis('off')


def facc(label, pred):
    pred = pred.ravel()
    label = label.ravel()
    return ((pred > 0.5) == label).mean()


if __name__ == '__main__':
    # =====================================设置训练参数==========================================================
    epochs = 2  # Set low by default for tests, set higher when you actually run this code.
    batch_size = 64
    latent_z_size = 100

    use_gpu = True
    ctx = mx.gpu() if use_gpu else mx.cpu()

    lr = 0.0002
    beta1 = 0.5
    # ===================================下载LFW人脸数据图像集===================================================
    lfw_url = 'http://vis-www.cs.umass.edu/lfw/lfw-deepfunneled.tgz'
    data_path = 'lfw_dataset'
    if not os.path.exists(data_path):
        os.makedirs(data_path)
        data_file = utils.download(lfw_url)
        with tarfile.open(data_file) as tar:
            tar.extractall(path=data_path)

    # ==================================对数据进行转换============================================================
    target_wd = 64
    target_ht = 64
    img_list = []
    for path, _, fnames in os.walk(data_path):
        for fname in fnames:
            if not fname.endswith('.jpg'):
                continue
            img = os.path.join(path, fname)
            img_arr = mx.image.imread(img)
            img_arr = transform(img_arr, target_wd, target_ht)
            img_list.append(img_arr)
    train_data = mx.io.NDArrayIter(data=nd.concatenate(img_list), batch_size=batch_size)

    # ========================================画出几幅图像================================================
    for i in range(4):
        plt.subplot(1, 4, i + 1)
        visualize(img_list[i + 10][0])
    plt.show()

    # ==========================================定义网络=========================================
    # build the generator
    nc = 3
    ngf = 64
    netG = nn.Sequential()
    with netG.name_scope():
        # input is Z, going into a convolution
        netG.add(nn.Conv2DTranspose(ngf * 8, 4, 1, 0, use_bias=False))
        netG.add(nn.BatchNorm())
        netG.add(nn.Activation('relu'))
        # state size. (ngf*8) x 4 x 4
        netG.add(nn.Conv2DTranspose(ngf * 4, 4, 2, 1, use_bias=False))
        netG.add(nn.BatchNorm())
        netG.add(nn.Activation('relu'))
        # state size. (ngf*8) x 8 x 8
        netG.add(nn.Conv2DTranspose(ngf * 2, 4, 2, 1, use_bias=False))
        netG.add(nn.BatchNorm())
        netG.add(nn.Activation('relu'))
        # state size. (ngf*8) x 16 x 16
        netG.add(nn.Conv2DTranspose(ngf, 4, 2, 1, use_bias=False))
        netG.add(nn.BatchNorm())
        netG.add(nn.Activation('relu'))
        # state size. (ngf*8) x 32 x 32
        netG.add(nn.Conv2DTranspose(nc, 4, 2, 1, use_bias=False))
        netG.add(nn.Activation('tanh'))
        # state size. (nc) x 64 x 64

    # build the discriminator
    ndf = 64
    netD = nn.Sequential()
    with netD.name_scope():
        # input is (nc) x 64 x 64
        netD.add(nn.Conv2D(ndf, 4, 2, 1, use_bias=False))
        netD.add(nn.LeakyReLU(0.2))
        # state size. (ndf) x 32 x 32
        netD.add(nn.Conv2D(ndf * 2, 4, 2, 1, use_bias=False))
        netD.add(nn.BatchNorm())
        netD.add(nn.LeakyReLU(0.2))
        # state size. (ndf) x 16 x 16
        netD.add(nn.Conv2D(ndf * 4, 4, 2, 1, use_bias=False))
        netD.add(nn.BatchNorm())
        netD.add(nn.LeakyReLU(0.2))
        # state size. (ndf) x 8 x 8
        netD.add(nn.Conv2D(ndf * 8, 4, 2, 1, use_bias=False))
        netD.add(nn.BatchNorm())
        netD.add(nn.LeakyReLU(0.2))
        # state size. (ndf) x 4 x 4
        netD.add(nn.Conv2D(1, 4, 1, 0, use_bias=False))

    # =================================================设置loss和optimizier===========================================
    loss = gluon.loss.SigmoidBinaryCrossEntropyLoss()

    # initialize the generator and the discriminator
    netG.initialize(mx.init.Normal(0.02), ctx=ctx)
    netD.initialize(mx.init.Normal(0.02), ctx=ctx)

    # trainer for the generator and the discriminator
    trainerG = gluon.Trainer(netG.collect_params(), 'adam', {'learning_rate': lr, 'beta1': beta1})
    trainerD = gluon.Trainer(netD.collect_params(), 'adam', {'learning_rate': lr, 'beta1': beta1})

    # ===============================================训练循环========================================================
    real_label = nd.ones((batch_size,), ctx=ctx)
    fake_label = nd.zeros((batch_size,), ctx=ctx)

    metric = mx.metric.CustomMetric(facc)

    stamp = datetime.now().strftime('%Y_%m_%d-%H_%M')
    logging.basicConfig(level=logging.DEBUG)

    for epoch in range(epochs):
        tic = time.time()
        btic = time.time()
        train_data.reset()
        iter = 0
        for batch in train_data:
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ############################
            data = batch.data[0].as_in_context(ctx)
            latent_z = mx.nd.random_normal(0, 1, shape=(batch_size, latent_z_size, 1, 1), ctx=ctx)

            with autograd.record():
                # train with real image
                output = netD(data).reshape((-1, 1))
                errD_real = loss(output, real_label)
                metric.update([real_label, ], [output, ])

                # train with fake image
                fake = netG(latent_z)
                output = netD(fake.detach()).reshape((-1, 1))
                errD_fake = loss(output, fake_label)
                errD = errD_real + errD_fake
                errD.backward()
                metric.update([fake_label, ], [output, ])

            trainerD.step(batch.data[0].shape[0])

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ############################
            with autograd.record():
                fake = netG(latent_z)
                output = netD(fake).reshape((-1, 1))
                errG = loss(output, real_label)
                errG.backward()

            trainerG.step(batch.data[0].shape[0])

            # Print log infomation every ten batches
            if iter % 10 == 0:
                name, acc = metric.get()
                logging.info('speed: {} samples/s'.format(batch_size / (time.time() - btic)))
                logging.info(
                    'discriminator loss = %f, generator loss = %f, binary training acc = %f at iter %d epoch %d'
                    % (nd.mean(errD).asscalar(),
                       nd.mean(errG).asscalar(), acc, iter, epoch))
            iter = iter + 1
            btic = time.time()

        name, acc = metric.get()
        metric.reset()
        # logging.info('\nbinary training acc at epoch %d: %s=%f' % (epoch, name, acc))
        # logging.info('time: %f' % (time.time() - tic))

        # Visualize one generated image for each epoch
        # fake_img = fake[0]
        # visualize(fake_img)
        # plt.show()

    # =================================================结果=========================================================
    num_image = 8
    for i in range(num_image):
        latent_z = mx.nd.random_normal(0, 1, shape=(1, latent_z_size, 1, 1), ctx=ctx)
        img = netG(latent_z)
        plt.subplot(2, 4, i + 1)
        visualize(img[0])
    plt.show()

    num_image = 12
    latent_z = mx.nd.random_normal(0, 1, shape=(1, latent_z_size, 1, 1), ctx=ctx)
    step = 0.05
    for i in range(num_image):
        img = netG(latent_z)
        plt.subplot(3, 4, i + 1)
        visualize(img[0])
        latent_z += 0.05
    plt.show()
    pass