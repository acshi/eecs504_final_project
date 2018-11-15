#!/usr/bin/python3

import time
import random
import torch
from torch import nn, optim
from torch.autograd.variable import Variable
from torchvision import transforms, datasets
from utils import Logger

# Set random seem for reproducibility
manualSeed = 999
torch.manual_seed(manualSeed)
torch.set_num_threads(3)

def mnist_data():
    compose = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((.5, .5, .5), (.5, .5, .5))
        ])
    out_dir = './dataset'
    return datasets.MNIST(root=out_dir, train=True, transform=compose, download=True)

# Load data
data = mnist_data()

# Create loader with data, so that we can iterate over it
data_loader = torch.utils.data.DataLoader(data, batch_size=100, shuffle=True)
# Num batches
num_batches = len(data_loader)

class DiscriminatorNet(torch.nn.Module):
    """
    A three hidden-layer discriminative neural network
    """
    def __init__(self):
        super(DiscriminatorNet, self).__init__()
        n_features = 784
        n_out = 1

        # self.hidden0 = nn.Sequential(
        #     nn.Linear(n_features, 1024),
        #     nn.LeakyReLU(0.2),
        #     nn.Dropout(0.3)
        # )
        # self.hidden1 = nn.Sequential(
        #     nn.Linear(1024, 512),
        #     nn.LeakyReLU(0.2),
        #     nn.Dropout(0.3)
        # )
        # self.hidden2 = nn.Sequential(
        #     nn.Linear(512, 256),
        #     nn.LeakyReLU(0.2),
        #     nn.Dropout(0.3)
        # )
        # self.out = nn.Sequential(
        #     nn.Linear(256, n_out),
        #     nn.Sigmoid()
        # )

        self.stage1 = nn.Sequential(
            nn.Conv2d(1, 64, 4, stride=2, padding=1, bias=False),
            # nn.BatchNorm2d(64), # normalize features to allow faster training. Also regularizes
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.1),

            nn.Conv2d(64, 64*2, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.1),
        )

        self.dicriminate = nn.Sequential(
            self.stage1,
            nn.Conv2d(64*2, 1, 7, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )

        self.classify = nn.Sequential(
            self.stage1,
            nn.Conv2d(64*2, 10, 7, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )

    def freeze_stage1(self):
        for p in self.stage1.parameters():
            p.requires_grad = False

    def unfreeze_stage1(self):
        for p in self.stage1.parameters():
            p.requires_grad = True

    def forward(self, x):
        # x = self.hidden0(x)
        # x = self.hidden1(x)
        # x = self.hidden2(x)
        # x = self.out(x)
        # return x
        return self.dicriminate(x)

discriminator = DiscriminatorNet()

# def images_to_vectors(images):
#     return images.view(images.size(0), 784)
#
# def vectors_to_images(vectors):
#     return vectors.view(vectors.size(0), 1, 28, 28)

class GeneratorNet(torch.nn.Module):
    """
    A three hidden-layer generative neural network
    """
    def __init__(self):
        super(GeneratorNet, self).__init__()
        n_features = 100
        n_out = 784

        # Based on https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
        self.hidden = nn.Sequential(
            # We start out with a 1 "pixel" "image" with n_features "channels"
            nn.ConvTranspose2d(n_features, 64*2, 7, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(64*2),
            nn.ReLU(True),
            nn.Dropout(0.1), #0.1-0.2 works better for conv nets. (randomly zero 10% of features)
            # 7x7

            nn.ConvTranspose2d(64*2, 64, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Dropout(0.1),
            # 14x14

            nn.ConvTranspose2d(64, 1, 4, stride=2, padding=1, bias=False),
            nn.Tanh(),
            # 28x28
        )

        # self.hidden0 = nn.Sequential(
        #     nn.Linear(n_features, 256),
        #     nn.LeakyReLU(0.2)
        # )
        # self.hidden1 = nn.Sequential(
        #     nn.Linear(256, 512),
        #     nn.LeakyReLU(0.2)
        # )
        # self.hidden2 = nn.Sequential(
        #     nn.Linear(512, 1024),
        #     nn.LeakyReLU(0.2)
        # )
        #
        # self.out = nn.Sequential(
        #     nn.Linear(1024, n_out),
        #     nn.Tanh()
        # )

    def forward(self, x):
        # x = self.hidden0(x)
        # x = self.hidden1(x)
        # x = self.hidden2(x)
        # x = self.out(x)
        # return x
        return self.hidden(x)

generator = GeneratorNet()

def noise(size):
    '''
    Generates a 1-d vector of gaussian sampled random values
    '''
    n = Variable(torch.randn(size, 100, 1, 1))
    return n

print("Has CUDA? {}".format(torch.cuda.is_available()))
print("Using {} threads!".format(torch.get_num_threads()))
print("Discriminator parameter count: {}".format(sum(p.numel() for p in discriminator.parameters() if p.requires_grad)))
print("Generator parameter count: {}".format(sum(p.numel() for p in generator.parameters() if p.requires_grad)))
# exit()

d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0004)
g_optimizer = optim.Adam(generator.parameters(), lr=0.0004)

loss = nn.BCELoss()

def ones_target(size):
    '''
    Tensor containing ones, with shape = size
    '''
    data = Variable(torch.ones(size, 1, 1, 1) * 1.2 - torch.rand(size, 1, 1, 1) * 0.4)
    return data

def zeros_target(size):
    '''
    Tensor containing zeros, with shape = size
    '''
    data = Variable(torch.rand(size, 1, 1, 1) * 0.4)
    return data

def train_discriminator(optimizer, real_data, N):
    # 1.1 Train on Real Data
    prediction_real = discriminator(real_data)

    # Reset gradients
    optimizer.zero_grad()
    # Calculate error and backpropagate
    error_real = loss(prediction_real, ones_target(N))
    error_real.backward()

    # Generate fake data and detach
    # (so gradients are not calculated for generator)
    fake_data = generator(noise(N)).detach()

    # 1.2 Train on Fake Data
    prediction_fake = discriminator(fake_data)
    # Calculate error and backpropagate
    error_fake = loss(prediction_fake, zeros_target(N))
    error_fake.backward()

    # 1.3 Update weights with gradients
    optimizer.step()

    # Return error and predictions for real and fake inputs
    return error_real + error_fake, prediction_real, prediction_fake

def labels_to_onehot(real_labels):
    batch_size = real_labels.shape[0]
    one_hot = torch.zeros([batch_size, 10, 1, 1])
    for i in range(0, batch_size):
        one_hot[i, real_labels[i], 0, 0] = 1

    return one_hot

def train_classifier(optimizer, real_data, real_labels):
    discriminator.freeze_stage1()

    prediction = discriminator.classify(real_data)

    optimizer.zero_grad()
    error_classify = loss(prediction, labels_to_onehot(real_labels))
    error_classify.backward()

    discriminator.unfreeze_stage1()

    return error_classify

def train_generator(optimizer, N):
    fake_data = generator(noise(N))
    prediction = discriminator(fake_data)

    # Reset gradients, they are filled by the call to backward
    optimizer.zero_grad()

    # Calculate error and backpropagate
    error = loss(prediction, ones_target(N))
    error.backward()

    # Update weights with gradients
    optimizer.step()

    # Return error
    return error

num_test_samples = 16
test_noise = noise(num_test_samples)

# Create logger instance
logger = Logger(model_name='VGAN', data_name='MNIST')

# Total number of epochs to train
num_epochs = 200

batch_start_time = time.time()
for epoch in range(num_epochs):
    for n_batch, (real_batch, real_labels) in enumerate(data_loader):
        N = real_batch.size(0)

        # 1. Train Discriminator
        real_data = Variable(real_batch)
        d_error, d_pred_real, d_pred_fake = train_discriminator(d_optimizer, real_data, N)

        # 2. Train Generator
        g_error = train_generator(g_optimizer, N)

        # 3. Train Classifier
        c_error = train_classifier(d_optimizer, real_data, real_labels)

        # Log batch error
        logger.log(d_error, g_error, epoch, n_batch, num_batches)

        # Display Progress every few batches
        if n_batch % 10 == 0:
            now_time = time.time()
            elapsed = now_time - batch_start_time
            batch_start_time = now_time
            print("Batches took {:.3f} ms".format(elapsed * 1000))

            test_images = generator(test_noise)
            test_images = test_images.data

            logger.log_images(
                test_images, num_test_samples,
                epoch, n_batch, num_batches
            );
            # Display status Logs
            logger.display_status(
                epoch, num_epochs, n_batch, num_batches,
                d_error, g_error, c_error, d_pred_real, d_pred_fake
            )
