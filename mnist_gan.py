#!/usr/bin/python3

import time
import random
import os

# supress silly warnings from master branch of tensorflow...
import sys
sys.stderr = None

import torch
import torchvision
from utils import Logger

import tensorflow as tf
from tensorflow import nn, layers
import numpy as np

# Set random seem for reproducibility
manualSeed = 999
np.random.seed(manualSeed)
torch.manual_seed(manualSeed)
tf.set_random_seed(manualSeed)

def cifar_data():
    compose = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(64),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((.5, .5, .5), (.5, .5, .5))
        ])
    out_dir = './dataset/cifar'
    return torchvision.datasets.CIFAR10(root=out_dir, train=True, transform=compose, download=True)

def mnist_data():
    compose = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((.5, .5, .5), (.5, .5, .5))
        ])
    out_dir = './dataset/mnist'
    return torchvision.datasets.MNIST(root=out_dir, train=True, transform=compose, download=True)

batch_size = 100
gen_features = 100

# Load data
data = mnist_data()
image_size = (28, 28, 1)
# data = cifar_data()
# image_size = (64, 64, 3)

# Create loader with data, so that we can iterate over it
data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)
num_batches = len(data_loader)

keep_prob = tf.placeholder(tf.float32, name='keep_prob')

def noise(size):
    return np.random.normal(size=size)

def discriminator(x):
    with tf.variable_scope("discriminator", reuse=tf.AUTO_REUSE):
        with tf.variable_scope("conv1"):
            conv1 = layers.conv2d(x, 64, 4, strides=2, padding="same", use_bias=False)
            conv1 = layers.batch_normalization(conv1, scale=False)
            conv1 = nn.leaky_relu(conv1, 0.2)
            conv1 = nn.dropout(conv1, keep_prob)

        with tf.variable_scope("conv2"):
            conv2 = layers.conv2d(conv1, 64*2, 4, strides=2, padding="same", use_bias=False)
            conv2 = layers.batch_normalization(conv2, scale=False)
            conv2 = nn.leaky_relu(conv2, 0.2)
            conv2 = nn.dropout(conv2, keep_prob)

        with tf.variable_scope("linear"):
            linear = layers.flatten(conv2)
            linear = layers.dense(linear, 1, use_bias=False)

        with tf.variable_scope("out"):
            out = nn.sigmoid(linear)
    return out

def generator(z):
    with tf.variable_scope("generator", reuse=tf.AUTO_REUSE):
        with tf.variable_scope("conv1_transp"):
            conv1 = tf.reshape(z, (-1, 1, 1, gen_features))
            conv1 = layers.conv2d_transpose(conv1, 64 * 2, 7, strides=1, use_bias=False)
            conv1 = layers.batch_normalization(conv1)
            conv1 = nn.relu(conv1)
            conv1 = nn.dropout(conv1, keep_prob)

        with tf.variable_scope("conv2_transp"):
            conv2 = layers.conv2d_transpose(conv1, 64, 4, strides=2, padding="same", use_bias=False)
            conv2 = layers.batch_normalization(conv2)
            conv2 = nn.relu(conv2)
            conv2 = nn.dropout(conv2, keep_prob)

        with tf.variable_scope("conv3_transp"):
            conv3 = layers.conv2d_transpose(conv2, 1, 4, strides=2, padding="same", use_bias=False)

        with tf.variable_scope("out"):
            out = tf.tanh(conv3)
    return out

## Real Input
X = tf.placeholder(tf.float32, shape=(None, ) + image_size)
## Latent Variables / Noise
Z = tf.placeholder(tf.float32, shape=(None, gen_features))

# Generator
G_sample = generator(Z)
# Discriminator
D_real = discriminator(X)
D_fake = discriminator(G_sample)

# Generator
G_loss = tf.reduce_mean(
    nn.sigmoid_cross_entropy_with_logits(
        logits=D_fake, labels=tf.ones_like(D_fake) * 1.2 - tf.random.uniform(tf.shape(D_fake)) * 0.4
    )
)

# Discriminator
D_loss_real = tf.reduce_mean(
    nn.sigmoid_cross_entropy_with_logits(
        logits=D_real, labels=tf.ones_like(D_real) * 1.2 - tf.random.uniform(tf.shape(D_real)) * 0.4
    )
)

D_loss_fake = tf.reduce_mean(
    nn.sigmoid_cross_entropy_with_logits(
        logits=D_fake, labels=tf.random.uniform(tf.shape(D_fake)) * 0.4
    )
)

D_loss = D_loss_real + D_loss_fake

# Obtain trainable variables for both networks
train_vars = tf.trainable_variables()

G_vars = [var for var in train_vars if 'generator' in var.name]
D_vars = [var for var in train_vars if 'discriminator' in var.name]

print("Discriminator parameter count: {}".format(np.sum([np.product(v.get_shape()) for v in D_vars])))
print("Generator parameter count: {}".format(np.sum([np.product(v.get_shape()) for v in G_vars])))

G_opt = tf.train.AdamOptimizer(4e-4).minimize(G_loss, var_list=G_vars)
D_opt = tf.train.AdamOptimizer(4e-4).minimize(D_loss, var_list=D_vars)

num_test_samples = 16
test_noise = noise((num_test_samples, gen_features))

# Create logger instance
logger = Logger(model_name='DCGAN1')

# Total number of epochs to train
num_epochs = 200

# Start interactive session
session = tf.InteractiveSession()
# Init Variables
tf.global_variables_initializer().run()

batch_start_time = time.time()
for epoch in range(num_epochs):
    for n_batch, (real_images_torch, real_labels_torch) in enumerate(data_loader):
        # 1. Train Discriminator
        real_images = real_images_torch.permute(0, 2, 3, 1).numpy()
        feed_dict = {X: real_images, Z: noise((batch_size, gen_features)), keep_prob: 0.1}
        _, d_error, d_pred_real, d_pred_fake = session.run([D_opt, D_loss, D_real, D_fake], feed_dict=feed_dict)

        # 2. Train Generator
        feed_dict = {Z: noise((batch_size, gen_features)), keep_prob: 0.1}
        _, g_error = session.run([G_opt, G_loss], feed_dict=feed_dict)

        # Display Progress every few batches
        if n_batch % 10 == 0:
            now_time = time.time()
            elapsed = now_time - batch_start_time
            batch_start_time = now_time
            print("Batches took {:.3f} ms".format(elapsed * 1000))

            test_images = session.run(G_sample, feed_dict={Z: test_noise, keep_prob: 1.0})

            logger.log_images(
                test_images, num_test_samples,
                epoch, n_batch, num_batches
            )
            # Display status Logs
            logger.display_status(
                epoch, num_epochs, n_batch, num_batches,
                d_error, g_error, 0, d_pred_real, d_pred_fake
            )
