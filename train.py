import os

import torch

import torchvision.transforms as transforms

from src.dataloader import get_datasets
from src.losses import Loss
from src.model import ZeroDCE

# ---------------------------------------- DATA --------------------------------------- #

DATA_DIR = 'data\\lol_dataset'
TRAIN_DIR_NAME = 'our485'
TEST_DIR_NAME = 'eval15'

IMAGES_TRANSFORM = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize([256, 256]),
        # transforms.CenterCrop(224),
        # transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
])
# ---------------------------------------- TRAIN -------------------------------------- #
BATCH_SIZE = 64
NUM_EPOCHS = 1

# ---------------------------------------- MODEL -------------------------------------- #
MODEL_CONFIG = {
    ZeroDCE.INPUT_SIZE: 256,
    ZeroDCE.LAYERS_NUM: 7,
    ZeroDCE.LAYERS_WIDTH: 32,
    ZeroDCE.ITERATIONS_NUM: 8
}


def train(config):

    # TODO 1 All global configurable to a YAML configuration.
    # TODO 2 Wire losses and implement them.

    # Model
    model = ZeroDCE(config=MODEL_CONFIG)

    # loss function and optimizer
    loss_fn = Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Data
    train_data, test_data = get_datasets(data_dir=DATA_DIR,
                                         train_dir_name=TRAIN_DIR_NAME, test_dir_name=TEST_DIR_NAME,
                                         batch_size=BATCH_SIZE,
                                         transform=IMAGES_TRANSFORM)

    for epoch_num in range(NUM_EPOCHS):
        for batch_num, (train_low_light, train_high_light) in enumerate(iter(train_data)):

            # feedforward and calculate loss
            train_low_light_enhanced = model(train_low_light)
            loss = loss_fn(train_low_light, train_low_light_enhanced)

            # backward pass and update weights
            optimizer.zero_grad()
            # loss.backward() TODO Needed?

            optimizer.step()


if __name__ == '__main__':
    train(None)