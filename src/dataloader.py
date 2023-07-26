import os

from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_image


class LEDataset(Dataset):
    LOW_LIGHT_DIR_NAME = 'low'
    ENHANCED_DIR_NAME = 'high'

    _IMAGE_TYPE = 'png'

    def __init__(self, img_dir, enhanced_img_dir, device, transform=None, target_transform=None):
        super(LEDataset, self).__init__()
        self.img_dir = img_dir
        self.enhanced_img_dir = enhanced_img_dir
        self.image_names = [path for path in os.listdir(self.img_dir) if path.endswith(self._IMAGE_TYPE)]

        self.transform = transform
        self.target_transform = target_transform

        self.device = device

    def __len__(self):
        return len(self.image_names)

    @staticmethod
    def read_image_with_index_from_dir(directory, file_name):
        img_path = os.path.join(directory, file_name)
        return Image.open(img_path)

    def __getitem__(self, idx):

        file_name = self.image_names[idx]

        image = self.read_image_with_index_from_dir(self.img_dir, file_name)
        enhanced_image = self.read_image_with_index_from_dir(self.enhanced_img_dir, file_name)

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            enhanced_image = self.target_transform(enhanced_image)

        # # TODO Make sure memory is freed and not overflown.
        # image.data = image.data.to(self.device)
        # enhanced_image.data = image.data.to(self.device)

        return image, enhanced_image


def get_datasets(data_dir, train_dir_name, test_dir_name, batch_size, transform, device, shuffle=True):

    # Initialize datasets of LOL dataset.
    datasets_dict = {}

    for dataset_dir in [train_dir_name, test_dir_name]:
        images_dir_path = os.path.join(data_dir, dataset_dir, LEDataset.LOW_LIGHT_DIR_NAME)
        enhanced_images_dir_path = os.path.join(data_dir, dataset_dir, LEDataset.ENHANCED_DIR_NAME)

        print(images_dir_path)
        print(enhanced_images_dir_path)

        datasets_dict[dataset_dir] = LEDataset(img_dir=images_dir_path, enhanced_img_dir=enhanced_images_dir_path,
                                               transform=transform, target_transform=transform,
                                               device=device)

    from torch.utils.data.dataloader import default_collate

    collate_fn = lambda x: tuple(x_.to(device) for x_ in default_collate(x))
    train_dataloader = DataLoader(datasets_dict[train_dir_name], batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    test_dataloader = DataLoader(datasets_dict[test_dir_name], batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)

    # import matplotlib.pyplot as plt
    #
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
