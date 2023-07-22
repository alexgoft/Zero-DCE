import torch


class Loss(torch.nn.Module):
    def __init__(self):
        super(Loss, self).__init__()

        # W_col and W_tvA are the weights of the losses.
        self._w_col = 1
        self._w_ilm = 1

    def forward(self, images, enhanced_images):

        # import matplotlib.pyplot as plt
        #
        # plt.imshow(images[0].detach().numpy().transpose(1, 2, 0))
        # plt.show()
        # plt.imshow(enhanced_images[0].detach().numpy().transpose(1, 2, 0))
        # plt.show()

        loss_spa = self._spatial_consistency_loss()
        loss_exp = self._exposure_control_loss()
        loss_col = self._color_constancy_loss()
        loss_ilm = self._illumination_smoothness_loss()

        total_loss = loss_spa + loss_exp + self._w_col * loss_col + self._w_ilm * loss_ilm

        return torch.tensor(total_loss)

    @staticmethod
    def _spatial_consistency_loss():
        return 0

    @staticmethod
    def _exposure_control_loss():
        return 0

    @staticmethod
    def _color_constancy_loss():
        return 0

    @staticmethod
    def _illumination_smoothness_loss():
        return 0
