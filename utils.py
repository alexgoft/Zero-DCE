import torch
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F


def get_device():
    # Device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f"[INFO] CUDA is available, using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("[INFO] CUDA is not available, using CPU...")

    return device


def display_images(images):
    if not isinstance(images, list):
        images = [images]

    fig, axs = plt.subplots(ncols=len(images), squeeze=False)
    for i, img in enumerate(images):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(img)
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    plt.show()
