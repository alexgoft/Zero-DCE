import torch

import torchvision.transforms as transforms

from PIL import Image
from torch import nn


class ZeroDCE(torch.nn.Module):
    LAYERS_NUM = 'layers_num'
    LAYERS_WIDTH = 'layers_width'
    ITERATIONS_NUM = 'iterations_num'
    INPUT_SIZE = 'input_size'  # Input layer have input shape of (INPUT_SIZE, INPUT_SIZE, 3)

    _RGB_CHANNELS = 3

    _STRIDE = (1, 1)
    _KERNEL_SHAPE = (3, 3)
    _PADDING = 1  # use padding 1 to keep same shape between convolutions.

    def __init__(self, config, device):
        super(ZeroDCE, self).__init__()

        self._device = device

        # Model params
        self._input_shape = config[self.INPUT_SIZE]
        self._layers_num = config[self.LAYERS_NUM]
        self._layers_width = config[self.LAYERS_WIDTH]
        self._iterations_num = config[self.ITERATIONS_NUM]

        self._layers = self._initialize_dce_net_layers()

        self._model = torch.nn.Sequential(*self._layers)
        print(self._model)

        self._init_weights()

    def _init_weights(self):
        for layer in self._model:
            if isinstance(layer, nn.Conv2d):
                torch.nn.init.xavier_uniform(layer.weight)
                layer.bias.data.fill_(0.01)

    def _initialize_dce_net_layers(self):

        layers = []

        relu = torch.nn.ReLU()
        tanh = torch.nn.Tanh()

        # Every layer is followed by RELU, and the last is followed by tanh
        for layer_num in range(self._layers_num - 1):
            out_layers = self._layers_width
            in_layers = self._RGB_CHANNELS if layer_num == 0 else self._layers_width

            # print((num_half_net_layers - i % num_half_net_layers) - 1)
            conv = torch.nn.Conv2d(
                in_channels=in_layers,
                out_channels=out_layers,
                padding=self._PADDING,
                kernel_size=self._KERNEL_SHAPE,
                stride=self._STRIDE,
                device=self._device,
            )

            layers.append(conv)
            layers.append(relu)

        # Add final layer that produce the Curve maps
        layers.append(torch.nn.Conv2d(
            in_channels=self._layers_width, out_channels=self._iterations_num * self._RGB_CHANNELS,
            padding=self._PADDING, kernel_size=self._KERNEL_SHAPE, stride=self._STRIDE
        ))
        layers.append(tanh)

        return layers

    @staticmethod
    def _light_enhancement_curve_function(prev_le, curr_alpha):
        """
        This function represent a Higher-Order Curve:

        LEn(x) = LEn−1(x) + αnLEn−1(x)(1 − LEn−1(x)) s.t LEn−1(x) I(x) when n=1

            LE(I(x); α) is the enhanced version of the given input I(x),
            α ∈ [−1, 1] is the trainable curve parameter
        """
        curr_le = prev_le + curr_alpha * (torch.square(prev_le) - prev_le)

        return curr_le

    def forward(self, x):

        input_image = x.detach().clone()

        # @@@@@@@@@@@@@@@@@@ DCE-Net @@@@@@@@@@@@@@@@@@ #
        # Create Curve maps for input image x.
        mid_results = []

        # First num_half_net_layers layers. Results will be connected to further layers.
        for layer_num in range(self._layers_num - 1):
            x = self._layers[layer_num](x)
            if layer_num % 2 != 0:  # Save intermediate results after activations.
                mid_results.append(x)

        # Skip connections layers.
        for layer_num in range(self._layers_num, (self._layers_num * 2) - 1):
            x = self._layers[layer_num](x)
            if layer_num % 2 != 0:  # Save intermediate results after activations.
                layer_num_skip = (2 * len(mid_results)) - (layer_num // 2) - 1
                x = x + mid_results[layer_num_skip]

        # Last layer that produces the curve maps.
        x = self._layers[-1](x)

        # @@@@@@@@@@@@@@@@ Iterations @@@@@@@@@@@@@@@@@ #
        alpha_maps = torch.split(x, split_size_or_sections=3, dim=1)

        le = input_image
        le_middle = None

        for i, alpha_i in enumerate(alpha_maps):
            le = self._light_enhancement_curve_function(prev_le=le, curr_alpha=alpha_i)
            if i == self._iterations_num // 2:
                le_middle = le

            # import matplotlib.pyplot as plt
            #
            # plt.title(f'Iteration: 8{i + 1}')
            # plt.imshow(le.to('cpu').detach().numpy()[0].transpose(1, 2, 0))
            # plt.show()

        return le, le_middle
