import itertools
import torch

from torchvision.transforms import Grayscale


class Loss(torch.nn.Module):
    _EXP_LOSS_WINDOW = (16, 16)
    _WELL_EXPOSENESS_LEVEL = 0.6

    def __init__(self, device):
        super(Loss, self).__init__()

        # W_col and W_tvA are the weights of the losses.
        self._w_col = 1
        self._w_ilm = 1

        self._grayscale_transform = Grayscale()

        self.device = device

    def forward(self, images, enhanced_images):
        # import matplotlib.pyplot as plt
        #
        # plt.imshow(images[0].detach().numpy().transpose(1, 2, 0))
        # plt.show()
        # plt.imshow(enhanced_images[0].detach().numpy().transpose(1, 2, 0))
        # plt.show()

        loss_spa = self._spatial_consistency_loss()
        loss_exp = 10 * self._exposure_control_loss(input_batch=enhanced_images)
        loss_col = 5 * self._color_constancy_loss(input_batch=enhanced_images)
        loss_ilm = self._illumination_smoothness_loss()

        total_loss = loss_spa + loss_exp + loss_col * self._w_col + loss_ilm * self._w_ilm

        return torch.tensor(total_loss), {
            self._spatial_consistency_loss.__name__: loss_spa,
            self._exposure_control_loss.__name__: loss_exp,
            self._color_constancy_loss.__name__: loss_col,
            self._illumination_smoothness_loss.__name__: loss_ilm,
        }

    def _exposure_control_loss(self, input_batch):
        """
        loss to control the exposure level.
        """
        # TODO Is not correct?
        gray_scale = self._grayscale_transform(input_batch)

        patches = split_image(image=gray_scale, patch_size=self._EXP_LOSS_WINDOW)
        patches_mean = torch.mean(patches, dim=[1, 2])
        patches_mean -= self._WELL_EXPOSENESS_LEVEL

        return patches_mean.abs().mean()

    def _color_constancy_loss(self, input_batch):
        """
        loss to correct the potential color deviations in the enhanced
        image and also build the relations among the three adjusted channels.
        """
        mean_per_channel = input_batch.mean(dim=[2, 3])  # 2,3 => h,w

        loss_sum_tensor = torch.zeros(input_batch.shape[0]).to(self.device)
        for ch_1, ch_2 in itertools.permutations([0, 1, 2], 2):  # 0/1/2 -> R/G/B
            loss_sum_tensor += torch.square(mean_per_channel[:, ch_1] - mean_per_channel[:, ch_2])

        return loss_sum_tensor.mean()

    @staticmethod
    def _spatial_consistency_loss():
        return 0

    @staticmethod
    def _illumination_smoothness_loss():
        return 0


# TODO Utilities module?
def split_image(image, patch_size):
    """
        Divide image to non-overlapping patches of size patch_size.

        image: (batch_size, channels, height, width)
        patch_size: (patch_height, patch_width)
    """
    patches = torch.nn.functional.unfold(image, kernel_size=patch_size, stride=patch_size)
    patches = patches.reshape(-1, *patch_size)

    return patches
