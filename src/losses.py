import torch


class Loss(torch.nn.Module):
    def __init__(self):
        super(Loss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        pass
