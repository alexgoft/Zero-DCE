import torch


class Loss(torch.nn.Module):
    def __init__(self):
        super(Loss, self).__init__()

        # W_col and W_tvA are the weights of the losses.
        self._w_col = None
        self._w_ilm = None

    def forward(self, inputs):

        loss_spa = self._loss_spa()
        loss_exp = self._loss_exp()
        loss_col = self._loss_col()
        loss_ilm = self._loss_ilm()

        total_loss = loss_spa + loss_exp + self._w_col * loss_col + self._w_ilm * loss_ilm

        return total_loss

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
        return 9
