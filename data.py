import os
import cv2
import torch.cuda

from imutils import paths
from torch.utils.data import DataLoader, Dataset


class LEDataset(Dataset):

    def __init__(self,
                 image_paths,
                 # enhanced_img_paths,
                 transform=None, target_transform=None,
                 device='cpu'):
        super(LEDataset, self).__init__()
        self.image_paths = image_paths
        self.transform = transform
        self.device = device

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = read_image(self.image_paths[idx])
        if self.transform:
            image = self.transform(image)
        return image.to(self.device)


def read_image(image_path):
    """ Function to read image from path. """
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def get_dataset(dir_path, batch_size=4, transform_img=None, device='cpu'):
    """ function to create datasets from given paths."""
    image_paths = sorted(list(paths.list_images(dir_path)))
    dataset_cls = LEDataset(image_paths=image_paths, transform=transform_img, device=device)
    dataset = DataLoader(dataset_cls, batch_size=batch_size)

    print(f"[INFO] found {len(dataset)} examples in the dataset...")
    return dataset
