import os
import time
import torch

import torchvision.transforms as transforms

from src.data import get_datasets
from src.losses import Loss
from src.model import ZeroDCE
from src.model_orig import enhance_net_nopool

# ---------------------------------------- DATA --------------------------------------- #

DATA_DIR = 'data\\lol_dataset'
TRAIN_DIR_NAME = 'our485'
VAL_DIR_NAME = 'eval15'

RESIZE_SIZE = (512, 512)
IMAGES_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(RESIZE_SIZE, antialias=True),
    # transforms.CenterCrop(224),
    # transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
])
# ---------------------------------------- TRAIN -------------------------------------- #
BATCH_SIZE = 8
VALIDATION_SPLIT = 0.15
NUM_EPOCHS = 200
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 0.0001

# ---------------------------------------- MODEL -------------------------------------- #
MODEL_CONFIG = {
    ZeroDCE.INPUT_SIZE: 256,
    ZeroDCE.LAYERS_NUM: 7,
    ZeroDCE.LAYERS_WIDTH: 32,
    ZeroDCE.ITERATIONS_NUM: 8
}


# TODO 1 All global configurable to a YAML configuration.
# TODO 2 Add validation loop.

def train(config):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Configure output directory.
    output_dir_path = os.path.join('outputs', time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(output_dir_path)

    # Model
    # model = ZeroDCE(config=MODEL_CONFIG)
    model = enhance_net_nopool()
    model.to(device)
    model.train()

    print(model)

    # Datasets
    train_dir_path = os.path.join(DATA_DIR, TRAIN_DIR_NAME, 'low')
    val_dir_path = os.path.join(DATA_DIR, TRAIN_DIR_NAME, 'low')

    train_data, val_data = get_datasets(
        images_path=train_dir_path, gts_path=val_dir_path,
        transform_img=IMAGES_TRANSFORM, transform_ann=None,
        val_split=VALIDATION_SPLIT, batch_size=BATCH_SIZE, device=device
    )

    # loss function and optimizer
    loss_fn = Loss(device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    best_loss = float('inf')

    # Train Loop
    for epoch_num in range(NUM_EPOCHS):

        # model.train()
        # for batch_num, (train_low_light, train_high_light) in enumerate(train_data):
        #     # Move data to device
        #     train_low_light, train_high_light = train_low_light.to(device), train_high_light.to(device)
        #
        #     # feedforward and calculate loss
        #     train_low_light_enhanced = model(train_low_light)
        #     # _, train_low_light_enhanced, _ = model(train_low_light)
        #
        #     # Compute the loss and its gradients
        #     total_loss, losses_dict = loss_fn(train_low_light, train_low_light_enhanced)
        #     total_loss.backward()
        #
        #     # Backpropagation
        #     # Zero your gradients for every batch and Adjust learning weights
        #     optimizer.zero_grad()
        #     optimizer.step()
        #
        #     print(f'[INFO] EPOCH NUM: {epoch_num}, BATCH NUM: {batch_num}__{losses_dict}\n')

        # val_loss
        model.eval()
        val_loss = 0
        with torch.no_grad():

            # loop over the validation set
            for (test_low_light, _) in val_data:
                test_low_light = test_low_light.to(device)

                image_half_enhanced, image_enhanced, _ = model(test_low_light)

                total_loss, losses_dict = loss_fn(images_batch=image_enhanced, image_half_enhanced=image_half_enhanced)
                val_loss += total_loss.item()

        val_loss = val_loss / len(val_data)


if __name__ == '__main__':
    train(None)
