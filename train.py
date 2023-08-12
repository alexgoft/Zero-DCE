import torch

import torchvision.transforms as transforms

from torch.autograd import Variable
from tqdm import tqdm

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
BATCH_SIZE = 4
NUM_EPOCHS = 200

# ---------------------------------------- MODEL -------------------------------------- #
MODEL_CONFIG = {
    ZeroDCE.INPUT_SIZE: 256,
    ZeroDCE.LAYERS_NUM: 7,
    ZeroDCE.LAYERS_WIDTH: 32,
    ZeroDCE.ITERATIONS_NUM: 8
}


# TODO 1 All global configurable to a YAML configuration.
def train(config):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Model
    model = ZeroDCE(config=MODEL_CONFIG)
    model.to(device)
    model.train()

    # loss function and optimizer
    loss_fn = Loss(device=device)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0001 )

    # Data
    train_data, test_data = get_datasets(data_dir=DATA_DIR,
                                         train_dir_name=TRAIN_DIR_NAME, test_dir_name=TEST_DIR_NAME,
                                         batch_size=BATCH_SIZE,
                                         transform=IMAGES_TRANSFORM,
                                         device=device)

    # Train Loop
    for epoch_num in range(NUM_EPOCHS):
        for batch_num, (train_low_light, train_high_light) in (pbar := tqdm(enumerate(iter(train_data)))):

            # feedforward and calculate loss
            train_low_light_enhanced = model(train_low_light)

            # Compute the loss and its gradients
            total_loss, losses_dict = loss_fn(train_low_light, train_low_light_enhanced)
            total_loss.backward()

            # Backpropagation
            # Zero your gradients for every batch and Adjust learning weights
            optimizer.zero_grad()
            optimizer.step()

            # torch.nn.utils.clip_grad_norm(model.parameters(), 0.1)

            # pbar.set_description(f'EPOCH NUM: {epoch_num}, BATCH NUM: {batch_num}__{losses_dict}\n')
        print(losses_dict)


if __name__ == '__main__':
    train(None)
