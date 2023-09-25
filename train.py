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
    model = ZeroDCE(config=MODEL_CONFIG, device=device)
    # model = enhance_net_nopool()
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

    # Train Loop
    min_valid_loss = float('inf')
    for epoch_num in range(NUM_EPOCHS):

        train_loss = 0.0
        model.train()
        for batch_num, train_low_light in enumerate(train_data):
            print('[INFO]\tEPOCH NUM: {}, BATCH NUM: {}/{}'.format(epoch_num + 1, batch_num + 1, len(train_data)))

            train_low_light = train_low_light.to(device)

            # feedforward
            # image_enhanced, image_half_enhanced, _ = model(train_low_light)
            train_low_light_enhanced, image_half_enhanced = model(train_low_light)

            optimizer.zero_grad()

            loss, losses_dict = loss_fn(image_enhanced=train_low_light_enhanced, image_half_enhanced=image_half_enhanced)
            loss.backward()
            train_loss += loss.item()

            # Backpropagation
            optimizer.step()
        print(f'[INFO] EPOCH NUM: {epoch_num + 1}, TRAIN LOSS: {train_loss:.5f}')

        # val_loss
        model.eval()
        eval_loss = 0.0
        with torch.no_grad():

            # loop over the validation set
            for batch_num, eval_low_light_images in enumerate(eval_data):
                print('[INFO]\tEPOCH NUM: {}, BATCH NUM: {}/{}'.format(epoch_num + 1, batch_num + 1, len(eval_data)))

                eval_low_light_images = eval_low_light_images.to(device)

                image_half_enhanced, image_enhanced = model(eval_low_light_images)

                loss, losses_dict = loss_fn(image_enhanced=image_enhanced, image_half_enhanced=image_half_enhanced)
                eval_loss += loss.item()
        eval_loss = eval_loss / len(eval_data)
        print(f'[INFO] EPOCH NUM: {epoch_num + 1}, VALIDATION LOSS: {eval_loss:.5f}')

        if min_valid_loss > eval_loss:
            print(f'[INFO] Validation Loss Decreased({min_valid_loss:.5f}--->{eval_loss:.5f}) '
                  f'\t Saving The Model ({output_dir_path})')
            min_valid_loss = eval_loss

            # Saving State Dict
            model_output_path = os.path.join(output_dir_path, f'model_{eval_loss}.pth')
            torch.save(model.state_dict(), model_output_path)


if __name__ == '__main__':
    train(None)
