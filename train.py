import os
import time
import torch

import torchvision.transforms as transforms

from utils import get_device
from src.data import get_dataset
from src.losses import Loss
from src.model import ZeroDCE
from src.config_file import ConfigFile

# ---------------------------------------- DATA --------------------------------------- #

DATA_DIR = 'data\\lol_dataset'
TRAIN_DIR_NAME = 'our485'
VAL_DIR_NAME = 'eval15'

RESIZE_SIZE = (512, 512)
IMAGES_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(RESIZE_SIZE, antialias=True),
])

CONFIG_FILE_NAME = 'config.yaml'
CONFIG_FILE_PATH = os.path.join('src', CONFIG_FILE_NAME)


def train():
    device = get_device()

    # output directory
    output_dir_path = os.path.join('outputs', time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(output_dir_path)

    # config
    config = ConfigFile.load(config_path=CONFIG_FILE_PATH)

    output_dir_path = os.path.join(output_dir_path, 'config.yaml')
    config.save_config(output_dir_path)

    # model
    model = ZeroDCE(config=config, device=device)

    # Datasets
    train_dir_path = os.path.join(DATA_DIR, TRAIN_DIR_NAME)
    val_dir_path = os.path.join(DATA_DIR, VAL_DIR_NAME)

    # gts_path is empty string because we don't have ground truth images.
    batch_size = config.train.batch_size
    train_data = get_dataset(
        dir_path=train_dir_path, transform_img=IMAGES_TRANSFORM,
        batch_size=batch_size, device=device
    )
    eval_data = get_dataset(
        dir_path=val_dir_path, transform_img=IMAGES_TRANSFORM,
        batch_size=batch_size, device=device
    )
    # loss function and optimizer
    loss_fn = Loss(device=device)
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=config.train.learning_rate,
                                 weight_decay=config.train.weight_decay)

    # Train Loop
    min_valid_loss = float('inf')
    for epoch_num in range(config.train.num_epochs):

        model.train()
        train_loss = 0.0
        for batch_num, train_low_light in enumerate(train_data):
            print(f'[INFO] EPOCH NUM: {epoch_num + 1}, BATCH NUM: {batch_num + 1}/{len(train_data)}')
            enhanced, alpha_maps = model(train_low_light)

            loss, losses_dict = loss_fn(enhanced_images=enhanced,
                                        orig_images=train_low_light,
                                        alpha_maps=alpha_maps)
            train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # Calculate average loss over an epoch
        train_loss = train_loss / len(train_data)
        print(f'[INFO] EPOCH NUM: {epoch_num + 1}, TRAIN LOSS: {train_loss:.5f}')

        # val_loss
        model.eval()
        eval_loss = 0.0
        with torch.no_grad():
            print(f'[INFO] VALIDATION START')

            # loop over the validation set
            for batch_num, eval_low_light_images in enumerate(eval_data):
                enhanced, alpha_maps = model(eval_low_light_images)

                loss, losses_dict = loss_fn(enhanced_images=enhanced,
                                            orig_images=eval_low_light_images,
                                            alpha_maps=alpha_maps)
                eval_loss += loss.item()
        # Calculate average loss over an epoch # TODO Same as train_loss. Make function?
        eval_loss = round(eval_loss / len(eval_data), 5)
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
