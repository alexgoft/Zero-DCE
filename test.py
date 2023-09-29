import os

import torchvision.transforms as transforms

from src.data import get_dataset
from src.model import ZeroDCE
from utils import get_device, display_images

DATA_DIR = 'data\\lol_dataset'
TEST_DIR_NAME = 'small'

MODEL_PATH = "C:\\Users\\alexg\\Desktop\\projects\\Zero-DCE\\outputs\\20230927-201607\\model_0.0877.pth"
RESIZE_SIZE = (512, 512)
IMAGES_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(RESIZE_SIZE, antialias=True),
])

MODEL_CONFIG = {
    ZeroDCE.MODEL_PATH: MODEL_PATH,

    ZeroDCE.INPUT_SIZE: 256,
    ZeroDCE.LAYERS_NUM: 7,
    ZeroDCE.LAYERS_WIDTH: 32,
    ZeroDCE.ITERATIONS_NUM: 8
}


def test():
    device = get_device()

    # Model
    model = ZeroDCE(config=MODEL_CONFIG, device=device)

    # Dataset
    test_dir_path = os.path.join(DATA_DIR, TEST_DIR_NAME)
    test_data = get_dataset(
        dir_path=test_dir_path, transform_img=IMAGES_TRANSFORM, batch_size=1, device=device
    )

    for image in test_data:
        image = image.to(device=device)
        enhanced_image, alpha_maps = model(image)

        image, enhanced_image = image.cpu().squeeze(), enhanced_image.cpu().squeeze()
        display_images([image, enhanced_image])


if __name__ == '__main__':
    test()
