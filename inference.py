import os
import yaml

import torchvision.transforms as transforms

from config import ConfigFile
from src.data import get_dataset
from src.model import ZeroDCE
from utils import get_device, display_images

DATA_DIR = 'data\\lol_dataset'
TEST_DIR_NAME = 'small'

OUTPUTS_DIR = "C:\\Users\\alexg\\Desktop\\projects\\Zero-DCE\\outputs\\20230927-201607"
CONFIG_NAME = "config.yaml"
MODEL_NAME = "model_0.09017.pth"

RESIZE_SIZE = (512, 512)
IMAGES_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(RESIZE_SIZE, antialias=True),
])


def test():
    config_path = os.path.join(OUTPUTS_DIR, CONFIG_NAME)
    model_path = os.path.join(OUTPUTS_DIR, MODEL_NAME)
    test_dir_path = os.path.join(DATA_DIR, TEST_DIR_NAME)

    device = get_device()

    config = ConfigFile.load(config_path=config_path)
    model = ZeroDCE(model_path=model_path, config=config, device=device)

    # Dataset
    test_data = get_dataset(dir_path=test_dir_path,
                            transform_img=IMAGES_TRANSFORM, batch_size=1, device=device)

    for image in test_data:
        image = image.to(device=device)
        enhanced_image, alpha_maps = model(image)

        image, enhanced_image = image.cpu().squeeze(), enhanced_image.cpu().squeeze()
        display_images([image, enhanced_image])


if __name__ == '__main__':
    test()
