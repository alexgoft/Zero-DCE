import os
import cv2

from imutils import paths

from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from torchvision.transforms import transforms


class LEDataset(Dataset):

    def __init__(self,
                 image_paths, enhanced_img_paths,
                 transform=None, target_transform=None,
                 device='cpu'):
        super(LEDataset, self).__init__()
        self.image_paths = image_paths
        self.enhanced_img_paths = enhanced_img_paths

        self.transform = transform
        self.target_transform = target_transform

        self.device = device

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):

        image = read_image(self.image_paths[idx])
        enhanced_image = read_image(self.enhanced_img_paths[idx])

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            enhanced_image = self.target_transform(enhanced_image)

        return image, enhanced_image


def read_image(image_path):
    """ Function to read image from path. """
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def get_datasets(images_path, gts_path, transform_img, transform_ann, val_split, batch_size, device):
    """ Function to create train and test datasets and data loaders. """
    # Load paths in sorted manner.
    image_paths = sorted(list(paths.list_images(images_path)))
    gts_path = sorted(list(paths.list_images(gts_path)))

    # partition the data into training and testing splits using 85% of
    # the data for training and the remaining 15% for testing
    split = train_test_split(image_paths, gts_path, test_size=val_split)

    # unpack the data split
    (train_images, test_images) = split[:2]
    (train_enhanced_images, test_enhanced_images) = split[2:]

    # create the train and test datasets define transformations
    train_dataset = LEDataset(image_paths=test_images, enhanced_img_paths=test_enhanced_images,
                              transform=transform_img, target_transform=transform_ann, device=device)
    val_dataset = LEDataset(image_paths=test_images, enhanced_img_paths=test_enhanced_images,
                            transform=transform_img, target_transform=transform_ann, device=device)
    print(f"[INFO] found {len(train_dataset)} examples in the training set...")
    print(f"[INFO] found {len(val_dataset)} examples in the eval set...")

    # create the training and test data loaders
    pin_memory = True if 'cuda' in device.type else False
    train_loader = DataLoader(train_dataset,  batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset,  batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=pin_memory)

    return train_loader, val_loader
