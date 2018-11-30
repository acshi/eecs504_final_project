#!/usr/bin/python3

import time
import random
import os

# supress silly warnings from master branch of tensorflow...
# import sys
# sys.stderr = None

import matplotlib.image as mpimg
import skimage.transform

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

# make dictionary of lists of all bounding boxes
bbox_file = "WIDER_val/wider_face_split/wider_face_val_bbx_gt.txt"

min_image_size = 16
image_size_in = (16, 16, 3)
image_size_up = (64, 64, 3)

def data_generator():
    with open(bbox_file) as f:
        image_name = None
        bboxes_left = None
        current_image = None
        for line in f:
            line = line.strip()
            if image_name is None:
                image_name = line
                file_name = "WIDER_val/images/{}".format(image_name)
                current_image = mpimg.imread(file_name)
                current_image = np.array(current_image, dtype=np.float32) / 255.0
            elif bboxes_left is None:
                bboxes_left = int(line)
            else:
                numbers = [int(v) for v in line.split(" ")]
                x, y, w, h = numbers[0:4]

                bbox = numbers[0:4]

                # get sub-image and resize
                # ignore faces that are not already smaller than our small size
                if h > min_image_size and w > min_image_size:
                    sub_image = current_image[y:y+h, x:x+w]
                    full_image = skimage.transform.resize(sub_image, (image_size_up[0], image_size_up[1]), anti_aliasing=True, mode="constant")
                    small_image = skimage.transform.resize(sub_image, (image_size_in[0], image_size_in[1]), anti_aliasing=True, mode="constant")
                    full_image = np.array(full_image, dtype=np.float32)
                    small_image = np.array(small_image, dtype=np.float32)

                    yield (full_image, small_image, 1)

                    # then find a random part of the image for a non-face selection
                    total_h = current_image.shape[0]
                    total_w = current_image.shape[1]
                    h = image_size_up[0]
                    w = image_size_up[1]
                    x = random.randint(0, total_w - w)
                    y = random.randint(0, total_h - h)
                    full_image = current_image[y:y+h, x:x+w]
                    small_image = skimage.transform.resize(full_image, (image_size_in[0], image_size_in[1]), anti_aliasing=True, mode="constant")
                    small_image = np.array(small_image, dtype=np.float32)

                    yield (full_image, small_image, 0)

                bboxes_left -= 1
                if bboxes_left == 0:
                    image_name = None
                    bboxes_left = None
                    bboxes = []

def batch_generator(batch_size):
    data_gen = data_generator()
    images = []
    small_images = []
    labels = []
    batch_i = 0
    for (image, small_image, label) in data_gen:
        images += [image]
        small_images += [small_image]
        labels += [label]
        if len(images) == batch_size:
            yield (batch_i, (np.stack(images), np.stack(small_images), np.reshape(labels, [-1, 1])))
            images = []
            small_images = []
            labels = []
            batch_i += 1
# os.listdir("somedirectory")

batch_size = 100
num_batches = 39370 / batch_size # approximately

def noise(size):
    return np.random.normal(size=size)

def discriminator(x):
    with tf.variable_scope("discriminator", reuse=tf.AUTO_REUSE):
        with tf.variable_scope("conv1"):
            conv1 = layers.conv2d(x, 64, 3, strides=2, padding="same", activation=nn.relu)
            conv1 = layers.max_pooling2d(conv1, [2, 2], strides=2, padding="same")

        with tf.variable_scope("conv2"):
            conv2 = layers.conv2d(conv1, 128, 3, strides=2, padding="same", activation=nn.relu)
            conv2 = layers.max_pooling2d(conv2, [2, 2], strides=2, padding="same")

        with tf.variable_scope("conv3"):
            conv3 = layers.conv2d(conv2, 256, 3, strides=2, padding="same", activation=nn.relu)
            conv3 = layers.max_pooling2d(conv3, [2, 2], strides=2, padding="same")

        with tf.variable_scope("conv4"):
            conv4 = layers.conv2d(conv3, 512, 3, strides=2, padding="same", activation=nn.relu)
            conv4 = layers.max_pooling2d(conv4, [2, 2], strides=2, padding="same")

        with tf.variable_scope("conv5"):
            conv5 = layers.conv2d(conv4, 512, 3, strides=1, padding="same", activation=nn.relu)

        with tf.variable_scope("linear"):
            linear = layers.flatten(conv5)
            linear = layers.dense(linear, 2, use_bias=False)

        with tf.variable_scope("out"):
            out = nn.sigmoid(linear)
    return out

def generator(x):
    with tf.variable_scope("generator", reuse=tf.AUTO_REUSE):
        with tf.variable_scope("conv1"):
            conv1 = layers.conv2d(x, 64, 3, strides=1, padding="same")
            conv1 = layers.batch_normalization(conv1, scale=False)
            conv1 = nn.relu(conv1)

        with tf.variable_scope("conv2"):
            # residual block with (up to) 8 convolution layers
            n_conv_layers = 8;

            previous = conv1
            for i in range(n_conv_layers):
                conv2 = layers.conv2d(previous, 64, 3, strides=1, padding="same")
                conv2 = layers.batch_normalization(conv2, scale=False)
                conv2 += previous
                conv2 = nn.relu(conv2)
                previous = conv2

        with tf.variable_scope("conv3"):
            conv3 = layers.conv2d(conv2, 64, 3, strides=1, padding="same")
            conv3 = layers.batch_normalization(conv3, scale=False)
            conv3 = nn.relu(conv3)

        with tf.variable_scope("deconv4"):
            deconv4 = layers.conv2d_transpose(conv3, 256, 3, strides=2, padding="same")
            deconv4 = layers.batch_normalization(deconv4, scale=False)
            deconv4 = nn.relu(deconv4)

        with tf.variable_scope("deconv5"):
            deconv5 = layers.conv2d_transpose(deconv4, 256, 3, strides=2, padding="same")
            deconv5 = layers.batch_normalization(deconv5, scale=False)
            deconv5 = nn.relu(deconv5)

        with tf.variable_scope("conv6"):
            conv6 = layers.conv2d(deconv5, 3, 1, strides=1, padding="same")
            conv6 = nn.relu(conv6)

        with tf.variable_scope("out"):
            out = nn.tanh(conv6)
    return out

# real input (full size)
X = tf.placeholder(tf.float32, shape=(None, ) + image_size_up)
# real labels (face vs non-face)
X_labels = tf.placeholder(tf.float32, shape=(None, 1))
# downsized input image
Z = tf.placeholder(tf.float32, shape=(None, ) + image_size_in)

# Generator
G_sample = generator(Z)
# Discriminator, has two outputs [face (1.0) vs nonface (0.0), real (1.0) vs generated (0.0)]
D_real = discriminator(X)
D_real_face = tf.slice(D_real, [0, 0], [-1, 1])
D_real_real = tf.slice(D_real, [0, 1], [-1, 1])
D_fake = discriminator(G_sample)
D_fake_face = tf.slice(D_fake, [0, 0], [-1, 1])
D_fake_real = tf.slice(D_fake, [0, 1], [-1, 1])

# Generator, MSE pixel-wise loss
G_pixel_loss = tf.reduce_mean((G_sample - X)**2)
G_adversarial_loss = tf.reduce_mean(
    nn.sigmoid_cross_entropy_with_logits(
        logits=D_fake_real, labels=tf.ones_like(D_fake_real) # * 1.2 - tf.random.uniform(tf.shape(D_fake)) * 0.4
    )
)
G_classification_loss = tf.reduce_mean(
    nn.sigmoid_cross_entropy_with_logits(
        logits=D_fake_face, labels=X_labels
    )
)
G_loss = G_pixel_loss + 0.001 * G_adversarial_loss + 0.01 * G_classification_loss

# Discriminator
D_loss_real = tf.reduce_mean(
    nn.sigmoid_cross_entropy_with_logits(
        logits=D_real_real, labels=tf.ones_like(D_real_real)# * 1.2 - tf.random.uniform(tf.shape(D_real)) * 0.4
    )
)
D_loss_fake = tf.reduce_mean(
    nn.sigmoid_cross_entropy_with_logits(
        logits=D_fake_real, labels=tf.zeros_like(D_fake_real)# + tf.random.uniform(tf.shape(D_fake[:, 0])) * 0.4
    )
)
D_classification_loss_real = tf.reduce_mean(
    nn.sigmoid_cross_entropy_with_logits(
        logits=D_real_face, labels=X_labels
    )
)
D_classification_loss_fake = tf.reduce_mean(
    nn.sigmoid_cross_entropy_with_logits(
        logits=D_fake_face, labels=X_labels
    )
)
D_loss = D_loss_real + D_loss_fake + D_classification_loss_real + D_classification_loss_fake

# Obtain trainable variables for both networks
train_vars = tf.trainable_variables()

G_vars = [var for var in train_vars if 'generator' in var.name]
D_vars = [var for var in train_vars if 'discriminator' in var.name]

print("Discriminator parameter count: {}".format(np.sum([np.product(v.get_shape()) for v in D_vars])))
print("Generator parameter count: {}".format(np.sum([np.product(v.get_shape()) for v in G_vars])))

learning_rate = tf.placeholder(tf.float32, shape=[])
G_opt = tf.train.AdamOptimizer(learning_rate).minimize(G_loss, var_list=G_vars)
D_opt = tf.train.AdamOptimizer(learning_rate).minimize(D_loss, var_list=D_vars)

num_test_samples = 16
_, (test_batch, test_small_images, test_labels) = next(batch_generator(num_test_samples))

# Create logger instance
logger = Logger(model_name='FACEGAN')

# Total number of epochs to train
num_epochs = 10

# Start interactive session
session = tf.InteractiveSession()
# Init Variables
tf.global_variables_initializer().run()

batch_start_time = time.time()
for epoch in range(num_epochs):
    lr = 1e-4 if epoch < 5 else 1e-5

    batch_gen = batch_generator(batch_size)
    for n_batch, (real_images, small_images, real_labels) in batch_gen:
        # 1. Train Discriminator
        feed_dict = {X: real_images, X_labels: real_labels, Z: small_images, learning_rate: lr}
        _, d_error, d_pred_real, d_pred_fake = session.run([D_opt, D_loss, D_real, D_fake], feed_dict=feed_dict)

        # 2. Train Generator
        feed_dict = {X: real_images, X_labels: real_labels, Z: small_images, learning_rate: lr}
        _, g_error = session.run([G_opt, G_loss], feed_dict=feed_dict)

        # Display Progress every few batches
        if n_batch % 1 == 0:
            now_time = time.time()
            elapsed = now_time - batch_start_time
            batch_start_time = now_time
            print("Batches took {:.3f} ms".format(elapsed * 1000))

            test_images = session.run(G_sample, feed_dict={Z: test_small_images})

            logger.log_images(
                test_images, num_test_samples,
                epoch, n_batch, num_batches
            )
            # Display status Logs
            logger.display_status(
                epoch, num_epochs, n_batch, num_batches,
                d_error, g_error, 0, d_pred_real, d_pred_fake
            )
