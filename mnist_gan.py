#!/usr/bin/python3

import time
import random
import os

import torch
import torchvision
from utils import Logger

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import numpy as np

tf.enable_eager_execution()

# Set random seem for reproducibility
manualSeed = 999
np.random.seed(manualSeed)
torch.manual_seed(manualSeed)
tf.set_random_seed(manualSeed)

def mnist_data():
    compose = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor(),
         torchvision.transforms.Normalize((.5, .5, .5), (.5, .5, .5))
        ])
    out_dir = './dataset'
    return torchvision.datasets.MNIST(root=out_dir, train=True, transform=compose, download=True)

batch_size = 100
gen_features = 100

# Load data
data = mnist_data()
# Create loader with data, so that we can iterate over it
data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)
num_batches = len(data_loader)

discriminator_stage1 = models.Sequential([
    layers.Conv2D(64, 4, strides=2, padding="same", use_bias=False, input_shape=(28, 28, 1)),
    # layers.BatchNormalization(),
    layers.LeakyReLU(0.2),
    layers.Dropout(0.1),

    layers.Conv2D(64*2, 4, strides=2, padding="same", use_bias=False),
    layers.BatchNormalization(),
    layers.LeakyReLU(0.2),
    layers.Dropout(0.1),
])

discriminator = models.Sequential([
    discriminator_stage1,
    layers.Conv2D(1, 7, use_bias=False, activation="sigmoid"),
])
discriminator_compiled = tf.contrib.eager.defun(discriminator)

classifier = models.Sequential([
    discriminator_stage1,
    layers.Conv2D(10, 7, use_bias=False, activation="sigmoid")
])
classifier_compiled = tf.contrib.eager.defun(classifier)

generator = models.Sequential([
    # We start out with a 1 "pixel" "image" with gen_features "channels"
    layers.Conv2DTranspose(64*2, 7, padding="valid", use_bias=False, input_shape=(1, 1, gen_features)),
    layers.BatchNormalization(),
    layers.ReLU(),
    layers.Dropout(0.1), #0.1-0.2 works better for conv nets. (randomly zero 10% of features)
    # 7x7

    layers.Conv2DTranspose(64, 4, strides=2, padding="same", use_bias=False),
    layers.BatchNormalization(),
    layers.ReLU(),
    layers.Dropout(0.1),
    # 14x14

    layers.Conv2DTranspose(1, 4, strides=2, padding="same", use_bias=False, activation="tanh"),
    # 28x28
])
generator_compiled = tf.contrib.eager.defun(generator)

print("Discriminator parameter count: {}".format(discriminator.count_params()))
print("Generator parameter count: {}".format(generator.count_params()))
print("Additional classifier parameter count: {}".format(classifier.layers[1].count_params()))

def discriminator_real_loss(real_output):
    # real images should be labeled as '1'
    return tf.losses.sigmoid_cross_entropy(tf.ones_like(real_output), real_output)

def discriminator_fake_loss(generated_output):
    # fake images should be labeled as '0'
    return tf.losses.sigmoid_cross_entropy(tf.zeros_like(generated_output), generated_output)

def generator_loss(generated_output):
    # fake images should be labeled as '1'
    return tf.losses.sigmoid_cross_entropy(tf.ones_like(generated_output), generated_output)

def classifier_loss(classifier_output, real_labels):
    # real images should be labeled correctly
    return tf.losses.softmax_cross_entropy(tf.one_hot(real_labels, 10),
                                           tf.reshape(classifier_output, [-1, 10]))

d_optimizer = tf.train.AdamOptimizer(0.0002)
g_optimizer = tf.train.AdamOptimizer(0.0002)
c_optimizer = tf.train.AdamOptimizer(0.0002)

def train_discriminator(real_images):
    noise = tf.random_normal([batch_size, 1, 1, gen_features])
    generated_images = generator_compiled(noise, training=True)
    with tf.GradientTape() as grad_tape:
        real_output = discriminator_compiled(real_images, training=True)
        real_loss = discriminator_real_loss(real_output)

        generated_output = discriminator_compiled(generated_images, training=True)
        fake_loss = discriminator_fake_loss(generated_output)
        total_loss = real_loss + fake_loss
    gradients_of_d = grad_tape.gradient(total_loss, discriminator.variables)
    d_optimizer.apply_gradients(zip(gradients_of_d, discriminator.variables))

    return total_loss.numpy(), real_output, generated_output

def train_classifier(real_images, real_labels):
    with tf.GradientTape() as grad_tape:
        classifier_output = classifier_compiled(real_images, training=True)
        c_loss = classifier_loss(classifier_output, real_labels)
    gradients_of_c = grad_tape.gradient(c_loss, classifier.layers[1].variables)
    c_optimizer.apply_gradients(zip(gradients_of_c, classifier.layers[1].variables))
    return c_loss.numpy()

def train_generator(real_images):
    with tf.GradientTape() as grad_tape:
        noise = tf.random_normal([batch_size, 1, 1, gen_features])
        generated_images = generator_compiled(noise, training=True)
        generated_output = discriminator_compiled(generated_images, training=True)
        g_loss = generator_loss(generated_output)
    gradients_of_g = grad_tape.gradient(g_loss, generator.variables)
    g_optimizer.apply_gradients(zip(gradients_of_g, generator.variables))
    return g_loss.numpy()

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(g_optimizer=g_optimizer,
                                 d_optimizer=d_optimizer,
                                 c_optimizer=c_optimizer,
                                 generator=generator,
                                 discriminator=discriminator,
                                 classifier=classifier)

num_test_samples = 16
test_noise = tf.random_normal([num_test_samples, 1, 1, gen_features])

# Create logger instance
logger = Logger(model_name='VGAN', data_name='MNIST')

# Total number of epochs to train
num_epochs = 200

batch_start_time = time.time()
for epoch in range(num_epochs):
    for n_batch, (real_images_torch, real_labels_torch) in enumerate(data_loader):
        real_images = real_images_torch.permute(0, 2, 3, 1).numpy()
        real_labels = real_labels_torch.numpy()
        N = real_images.shape[0]

        d_loss, real_output, generated_output = train_discriminator(real_images)
        c_loss = train_classifier(real_images, real_labels)
        g_loss = train_generator(real_images)
        logger.log(d_loss, g_loss, epoch, n_batch, num_batches)

        # Display Progress every few batches
        if n_batch % 10 == 0:
            now_time = time.time()
            elapsed = now_time - batch_start_time
            batch_start_time = now_time
            print("Batches took {:.3f} ms".format(elapsed * 1000))

            test_images = generator_compiled(test_noise).numpy()

            logger.log_images(
                test_images, num_test_samples,
                epoch, n_batch, num_batches
            )
            # Display status Logs
            logger.display_status(
                epoch, num_epochs, n_batch, num_batches,
                d_loss, g_loss, c_loss, real_output.numpy(), generated_output.numpy()
            )
            # save progess
            checkpoint.save(file_prefix = checkpoint_prefix)
