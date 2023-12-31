import os

import torchvision.transforms as transforms

from config_file import ConfigFile
from data import get_dataset
from model import ZeroDCE
from utils import get_device, display_images

DATA_DIR = 'data\\lol_dataset\\'
TEST_DIR_NAME = 'alex'

OUTPUTS_DIR = "C:\\Users\\alexg\\Desktop\\projects\\Zero-DCE\\outputs\\20230930-192210"
CONFIG_NAME = "config.yaml"
MODEL_NAME = "model_0.163.pth"

RESIZE_SIZE = (512, 512)
IMAGES_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(RESIZE_SIZE, antialias=True),
])


def inference():
    config_path = os.path.join(OUTPUTS_DIR, CONFIG_NAME)
    model_path = os.path.join(OUTPUTS_DIR, MODEL_NAME)
    test_dir_path = os.path.join(DATA_DIR, TEST_DIR_NAME)

    device = get_device()

    config = ConfigFile.load(config_path=config_path)
    model = ZeroDCE(model_path=model_path, config=config, device=device)

    # Dataset
    test_data = get_dataset(dir_path=test_dir_path,
                            transform_img=IMAGES_TRANSFORM, batch_size=1, device=device)

    for i, image in enumerate(test_data):
        enhanced_image, alpha_maps = model(image)

        image, enhanced_image = image.cpu().squeeze(), enhanced_image.cpu().squeeze()
        display_images([image, enhanced_image], title=f"image_{i}")


if __name__ == '__main__':
    inference()
