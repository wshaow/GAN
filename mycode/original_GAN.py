# -*- coding: utf-8 -*-
"""
    创建人：wshaow
    创建时间：2018年8月9日
    最原始的GAN网络
    这里值得注意的几个点：
    1、生成网络与判别网络的更新次序，以及如何更新
    2、detach函数的使用
    3、mx.io.NDArrayIter的使用
"""

from __future__ import print_function
import matplotlib as mpl
from matplotlib import pyplot as plt
import mxnet as mx
from mxnet import gluon, autograd, nd
from mxnet.gluon import nn
import numpy as np

# set up logging
from datetime import datetime
import os
import time


def get_text_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
    'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]


if __name__ == '__main__':
    ctx = mx.cpu()
    X = nd.random_normal(shape=(1000, 2))
    A = nd.array([[1, 2], [-0.1, 0.5]])
    b = nd.array([1, 2])
    X = nd.dot(X, A) + b
    Y = nd.ones(shape=(1000, 1))

    # and stick them into an iterator
    batch_size = 4
    train_data = mx.io.NDArrayIter(X, Y, batch_size, shuffle=True)

    plt.scatter(X[:, 0].asnumpy(), X[:, 1].asnumpy())
    plt.show()
    print("The covariance matrix is")
    print(nd.dot(A, A.T))

    # build the generator
    netG = nn.Sequential()
    with netG.name_scope():
        netG.add(nn.Dense(2))  # 只有一个输出层

    # build the discriminator (with 5 and 3 hidden units respectively)
    netD = nn.Sequential()
    with netD.name_scope():
        netD.add(nn.Dense(5, activation='tanh'))
        netD.add(nn.Dense(3, activation='tanh'))
        netD.add(nn.Dense(2))   # 有两类， 0和1

    # loss
    loss = gluon.loss.SoftmaxCrossEntropyLoss()

    # initialize the generator and the discriminator
    netG.initialize(mx.init.Normal(0.02), ctx=ctx)
    netD.initialize(mx.init.Normal(0.02), ctx=ctx)

    # trainer for the generator and the discriminator
    trainerG = gluon.Trainer(netG.collect_params(), 'adam', {'learning_rate': 0.01})
    trainerD = gluon.Trainer(netD.collect_params(), 'adam', {'learning_rate': 0.05})

    real_label = mx.nd.ones((batch_size,), ctx=ctx)
    fake_label = mx.nd.zeros((batch_size,), ctx=ctx)
    metric = mx.metric.Accuracy()

    # 开始训练网络
    stamp = datetime.now().strftime('%Y_%m_%d-%H_%M')
    for epoch in range(10):
        tic = time.time()
        train_data.reset()  # 数据迭代器初始化到原来的情况
        for i, batch in enumerate(train_data):
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            # train with real_t
            data = batch.data[0].as_in_context(ctx)
            noise = nd.random_normal(shape=(batch_size, 2), ctx=ctx)

            with autograd.record():
                real_output = netD(data)
                errD_real = loss(real_output, real_label)

                fake = netG(noise)
                fake_output = netD(fake.detach())   # 注意这里detach的用意，也就是不要把生成网络的结构加入到netD中
                errD_fake = loss(fake_output, fake_label)
                errD = errD_real + errD_fake
                errD.backward()  # 这里可以缩进也可以不缩进

            trainerD.step(batch_size)
            metric.update([real_label, ], [real_output, ])
            metric.update([fake_label, ], [fake_output, ])

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            with autograd.record():
                output = netD(fake)
                errG = loss(output, real_label)  # 这里不用detach是因为不存在网络之间的连接的情况
                errG.backward()

            trainerG.step(batch_size)

        name, acc = metric.get()
        metric.reset()
        print('\n binary training acc at epoch %d: %s=%f' % (epoch, name, acc))
        print('time: %f' % (time.time() - tic))
        noise = nd.random_normal(shape=(100, 2), ctx=ctx)
        fake = netG(noise)
        plt.scatter(X[:, 0].asnumpy(), X[:, 1].asnumpy())
        plt.scatter(fake[:, 0].asnumpy(), fake[:, 1].asnumpy())
        plt.show()



