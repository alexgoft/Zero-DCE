import os
import time
import torch

import torchvision.transforms as transforms

from utils import get_device
from data import get_dataset
from losses import ZeroReferenceLoss
from model import ZeroDCE
from config_file import ConfigFile

# ---------------------------------------- DATA --------------------------------------- #

DATA_DIR = 'data\\lol_dataset'
TRAIN_DIR_NAME = 'our485'
VAL_DIR_NAME = 'eval15'

RESIZE_SIZE = (512, 512)
IMAGES_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(RESIZE_SIZE, antialias=True),
])

CONFIG_FILE_PATH = 'config.yaml'


# ---------------------------------------- TRAIN --------------------------------------- #
def evaluate_data(data, model, loss):
    """ Evaluate the model on a given dataset"""
    model.eval()
    eval_loss = 0.0
    with torch.no_grad():
        print(f'[INFO] VALIDATION START')

        # loop over the validation set
        for batch_num, eval_low_light_images in enumerate(data):
            enhanced, alpha_maps = model(eval_low_light_images)

            loss, losses_dict = loss(enhanced_images=enhanced,
                                     orig_images=eval_low_light_images,
                                     alpha_maps=alpha_maps)
            eval_loss += loss.item()

    # Calculate average loss over an epoch
    eval_loss = round(eval_loss / len(data), 5)
    return eval_loss


def train_epoch(epoch_num, train_data, model, optimizer, loss):
    """ Train the model for one epoch"""
    model.train()
    train_loss = 0.0
    for batch_num, train_low_light in enumerate(train_data):
        print(f'[INFO] EPOCH NUM: {epoch_num + 1}, BATCH NUM: {batch_num + 1}/{len(train_data)}')
        enhanced, alpha_maps = model(train_low_light)

        loss, losses_dict = loss(enhanced_images=enhanced,
                                 orig_images=train_low_light,
                                 alpha_maps=alpha_maps)
        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Calculate average loss over an epoch
    train_loss = train_loss / len(train_data)
    return train_loss

# ---------------------------------------- MAIN --------------------------------------- #

def train():
    device = get_device()

    # output directory
    output_dir_path = os.path.join('outputs', time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(output_dir_path)

    # config
    config = ConfigFile.load(config_path=CONFIG_FILE_PATH)
    config_path = os.path.join(output_dir_path, 'config.yaml')
    config.save_config(config_path)

    # model
    model = ZeroDCE(config=config, device=device)

    # Datasets
    batch_size = config.train.batch_size
    train_data = get_dataset(
        dir_path=os.path.join(DATA_DIR, TRAIN_DIR_NAME),
        transform_img=IMAGES_TRANSFORM, batch_size=batch_size, device=device
    )
    eval_data = get_dataset(
        dir_path=os.path.join(DATA_DIR, VAL_DIR_NAME),
        transform_img=IMAGES_TRANSFORM, batch_size=batch_size, device=device
    )
    # loss function and optimizer
    loss_fn = ZeroReferenceLoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=config.train.learning_rate,
                                 weight_decay=config.train.weight_decay)

    # Train Loop
    min_valid_loss = float('inf')
    for epoch_num in range(config.train.num_epochs):

        # Train the model for one epoch and calculate the train loss (average over epoch).
        train_loss = train_epoch(epoch_num, train_data, model, optimizer, loss_fn)
        print(f'[INFO] EPOCH NUM: {epoch_num + 1}, TRAIN LOSS: {train_loss:.5f}')

        # Evaluate the model on the validation set and calculate the validation loss.
        eval_loss = evaluate_data(eval_data, model, loss_fn)
        print(f'[INFO] EPOCH NUM: {epoch_num + 1}, VALIDATION LOSS: {eval_loss}')

        if min_valid_loss > eval_loss:
            print(f'[INFO] Validation Loss Decreased({min_valid_loss}--->{eval_loss}) '
                  f'\t Saving The Model ({output_dir_path})')
            min_valid_loss = eval_loss

            # Saving State Dict
            model_output_path = os.path.join(output_dir_path, f'model_{eval_loss}.pth')
            torch.save(model.state_dict(), model_output_path)
    print(f'[INFO] Training finished. Output directory: {output_dir_path}')

if __name__ == '__main__':
    train()
