import os

import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_image


class LEDataset(Dataset):
    LOW_LIGHT_DIR_NAME = 'low'
    ENHANCED_DIR_NAME = 'high'

    _IMAGE_TYPE = 'png'

    def __init__(self, img_dir, enhanced_img_dir, transform=None, target_transform=None):
        self.img_dir = img_dir
        self.enhanced_img_dir = enhanced_img_dir
        self.image_names = [path for path in os.listdir(self.img_dir) if path.endswith(self._IMAGE_TYPE)]

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.enhanced_img_dir)

    def get_files_in_path(self):
        pass

    @staticmethod
    def read_image_with_index_from_dir(directory, file_name):
        img_path = os.path.join(directory, file_name)

        return read_image(img_path)

    def __getitem__(self, idx):

        file_name = self.image_names[idx]

        image = self.read_image_with_index_from_dir(self.img_dir, file_name)
        enhanced_image = self.read_image_with_index_from_dir(self.enhanced_img_dir, file_name)

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            enhanced_image = self.target_transform(enhanced_image)

        return image, enhanced_image


def get_datasets(data_dir, train_dir_name, test_dir_name, batch_size):

    # TODO Add transforms

    # Initialize datasets of LOL dataset.
    datasets_dict = {}

    for dataset_type in [TRAIN_DIR_NAME, TEST_DIR_NAME]:
        images_dir_path = os.path.join(data_dir, dataset_type, LEDataset.LOW_LIGHT_DIR_NAME)
        enhanced_images_dir_path = os.path.join(data_dir, dataset_type, LEDataset.ENHANCED_DIR_NAME)

        print(images_dir_path)
        print(enhanced_images_dir_path)

        datasets_dict[dataset_type] = LEDataset(img_dir=images_dir_path, enhanced_img_dir=enhanced_images_dir_path)

    train_dataloader = DataLoader(datasets_dict[TRAIN_DIR_NAME], batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(datasets_dict[TEST_DIR_NAME], batch_size=batch_size, shuffle=True)

    # # Display image and label.
    # images_batch, enhanced_images_batch = next(iter(train_dataloader))
    #
    # img = images_batch[0].squeeze().permute([1, 2, 0])
    # enhanced_image = enhanced_images_batch[0].squeeze().permute([1, 2, 0])
    #
    # plt.imshow(img)
    # plt.show()
    #
    # plt.imshow(enhanced_image)
    # plt.show()

    return train_dataloader, test_dataloader


if __name__ == '__main__':
    DATA_DIR = '..\\data\\lol_dataset'

    TRAIN_DIR_NAME = 'our485'
    TEST_DIR_NAME = 'eval15'

    BATCH_SIZE = 64

    get_datasets(data_dir=DATA_DIR, train_dir_name=TRAIN_DIR_NAME, test_dir_name=TEST_DIR_NAME, batch_size=BATCH_SIZE)
