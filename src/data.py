import os
import cv2

from imutils import paths

from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from torchvision.transforms import transforms


class LEDataset(Dataset):

    def __init__(self,
                 image_paths,
                 # enhanced_img_paths,
                 transform=None, target_transform=None,
                 device='cpu'):
        super(LEDataset, self).__init__()
        self.image_paths = image_paths
        # self.enhanced_img_paths = enhanced_img_paths

        self.transform = transform
        self.target_transform = target_transform

        self.device = device

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):

        image = read_image(self.image_paths[idx])
        # enhanced_image = read_image(self.enhanced_img_paths[idx])

        if self.transform:
            image = self.transform(image)
        # if self.target_transform:
        #     enhanced_image = self.target_transform(enhanced_image)
        enhanced_image = None

        return image # , enhanced_image


def read_image(image_path):
    """ Function to read image from path. """
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def get_datasets(train_dir_path, eval_dir_path,
                 batch_size=4,
                 transform_img=None, transform_ann=None,
                 device='cpu'):
    """ Function to create train and test datasets and data loaders. """
    # Load paths in sorted manner.
    train_paths = sorted(list(paths.list_images(train_dir_path)))
    eval_paths = sorted(list(paths.list_images(eval_dir_path)))

    # create the train and test datasets define transformations
    train_dataset = LEDataset(image_paths=train_paths, transform=transform_img, device=device)
    val_dataset = LEDataset(image_paths=eval_paths, transform=transform_img, device=device)

    # create the training and test data loaders
    pin_memory = True if 'cuda' in str(device) else False
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=pin_memory)
    eval_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=pin_memory)

    print(f"[INFO] found {len(train_dataset)} examples in the training set...")
    print(f"[INFO] found {len(val_dataset)} examples in the eval set...")

    return train_loader, eval_loader
