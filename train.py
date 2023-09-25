import os
import time
import torch

import torchvision.transforms as transforms
from tqdm import tqdm

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
VALIDATION_SPLIT = 1  # 0.15
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
    train_dir_path = os.path.join(DATA_DIR, TRAIN_DIR_NAME)
    val_dir_path = os.path.join(DATA_DIR, VAL_DIR_NAME)

    # gts_path is empty string because we don't have ground truth images.
    train_data, eval_data = get_datasets(
        train_dir_path=train_dir_path, eval_dir_path=val_dir_path,
        transform_img=IMAGES_TRANSFORM, batch_size=BATCH_SIZE, device=device
    )

    # loss function and optimizer
    loss_fn = Loss(device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    best_loss = float('inf')

    # Train Loop
    for epoch_num in range(NUM_EPOCHS):

        train_loss = 0.0
        model.train()
        for batch_num, (train_low_light, train_high_light) in tqdm(enumerate(train_data), total=len(train_data)):

            optimizer.zero_grad()

            # Move data to device
            train_low_light, train_high_light = train_low_light.to(device), train_high_light.to(device)

            # feedforward and calculate loss
            train_low_light_enhanced = model(train_low_light)
            # _, train_low_light_enhanced, _ = model(train_low_light)

            # Compute the loss and its gradients
            loss, losses_dict = loss_fn(train_low_light, train_low_light_enhanced)
            loss.backward()
            train_loss += loss.item()

            # Backpropagation
            # Zero your gradients for every batch and Adjust learning weights
            optimizer.step()
        print('[INFO] EPOCH NUM: {}, TRAINING LOSS: {}'.format(epoch_num + 1, train_loss / len(train_data)))

        # val_loss
        model.eval()
        eval_loss = 0.0
        with torch.no_grad():

            # loop over the validation set
            for eval_low_light_images in tqdm(eval_data, total=len(eval_data)):
                eval_low_light_images = eval_low_light_images.to(device)

                image_half_enhanced, image_enhanced, _ = model(eval_low_light_images)

                loss, losses_dict = loss_fn(image_enhanced=image_enhanced, image_half_enhanced=image_half_enhanced)
                eval_loss += loss.item()
        eval_loss = eval_loss / len(eval_data)
        print('[INFO] EPOCH NUM: {}, VALIDATION LOSS: {}'.format(epoch_num + 1, eval_loss))


if __name__ == '__main__':
    train(None)
