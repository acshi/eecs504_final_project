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

classifier = models.Sequential([
    discriminator_stage1,
    layers.Conv2D(10, 7, use_bias=False, activation="sigmoid")
])

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

# print("Has CUDA? {}".format(torch.cuda.is_available()))
# print("Using {} threads!".format(torch.get_num_threads()))
print("Discriminator parameter count: {}".format(discriminator.count_params()))
print("Generator parameter count: {}".format(generator.count_params()))
print("Additional classifier parameter count: {}".format(classifier.layers[1].count_params()))

# discriminator.compile(optimizer=tf.train.AdamOptimizer(0.0004),
#                       loss='binary_crossentropy',
#                       metrics=['accuracy'])
#
# classifier.compile(optimizer=tf.train.AdamOptimizer(0.0004),
#                    loss='categorical_crossentropy',
#                    metrics=['accuracy'])
#
# generator.compile(optimizer=tf.train.AdamOptimizer(0.0004),
#                   loss='binary_crossentropy',
#                   metrics=['accuracy'])

def discriminator_loss(real_output, generated_output):
    # real images should be labeled as '1'
    real_loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=tf.ones_like(real_output), logits=real_output)
    # fake images should be labeled as '0'
    generated_loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=tf.zeros_like(generated_output), logits=generated_output)
    return real_loss + generated_loss

def generator_loss(generated_output):
    # fake images should be labeled as '1'
    return tf.losses.sigmoid_cross_entropy(tf.ones_like(generated_output), generated_output)

def classifier_loss(classifier_output, real_labels):
    # real images should be labeled correctly
    return tf.losses.sigmoid_cross_entropy(tf.one_hot(real_labels, 10),
                                           tf.reshape(classifier_output, [-1, 10]))

d_optimizer = tf.train.AdamOptimizer(0.0004)
g_optimizer = tf.train.AdamOptimizer(0.0004)
c_optimizer = tf.train.AdamOptimizer(0.0004)

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

        noise = tf.random_normal([batch_size, 1, 1, gen_features])

        with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape, tf.GradientTape() as c_tape:
            generated_images = generator(noise, training=True)

            real_output = discriminator(real_images, training=True)
            generated_output = discriminator(generated_images, training=True)
            classifier_output = classifier(real_images, training=True)

            g_loss = generator_loss(generated_output)
            d_loss = discriminator_loss(real_output, generated_output)
            c_loss = classifier_loss(classifier_output, real_labels)

        gradients_of_g = g_tape.gradient(g_loss, generator.variables)
        gradients_of_d = d_tape.gradient(d_loss, discriminator.variables)
        gradients_of_c = c_tape.gradient(c_loss, classifier.layers[1].variables)

        g_optimizer.apply_gradients(zip(gradients_of_g, generator.variables))
        d_optimizer.apply_gradients(zip(gradients_of_d, discriminator.variables))
        c_optimizer.apply_gradients(zip(gradients_of_c, classifier.layers[1].variables))

        logger.log(d_loss.numpy(), g_loss.numpy(), epoch, n_batch, num_batches)

        # Display Progress every few batches
        if n_batch % 10 == 0:
            now_time = time.time()
            elapsed = now_time - batch_start_time
            batch_start_time = now_time
            print("Batches took {:.3f} ms".format(elapsed * 1000))

            test_images = generator(test_noise).numpy()

            logger.log_images(
                test_images, num_test_samples,
                epoch, n_batch, num_batches
            )
            # Display status Logs
            logger.display_status(
                epoch, num_epochs, n_batch, num_batches,
                d_loss.numpy(), g_loss.numpy(), c_loss.numpy(), real_output.numpy(), generated_output.numpy()
            )
            # save progess
            checkpoint.save(file_prefix = checkpoint_prefix)
