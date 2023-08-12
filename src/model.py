import torch

import torchvision.transforms as transforms

from PIL import Image


class ZeroDCE(torch.nn.Module):
    LAYERS_NUM = 'layers_num'
    LAYERS_WIDTH = 'layers_width'
    ITERATIONS_NUM = 'iterations_num'
    INPUT_SIZE = 'input_size'  # Input layer have input shape of (INPUT_SIZE, INPUT_SIZE, 3)

    _RGB_CHANNELS = 3

    _STRIDE = (1, 1)
    _KERNEL_SHAPE = (3, 3)
    _PADDING = 1  # use padding 1 to keep same shape between convolutions.

    def __init__(self, config):
        super(ZeroDCE, self).__init__()

        # Model params
        self._input_shape = config[self.INPUT_SIZE]
        self._layers_num = config[self.LAYERS_NUM]
        self._layers_width = config[self.LAYERS_WIDTH]
        self._iterations_num = config[self.ITERATIONS_NUM]

        self._layers = self._initialize_dce_net_layers()

        self._model = torch.nn.Sequential(*self._layers)
        print(self._model)

        # TODO Initialize model weights as specified in the paper.

    def _initialize_dce_net_layers(self):

        layers = []

        relu = torch.nn.ReLU()
        tanh = torch.nn.Tanh()

        # Every layer is followed by RELU, and the last is followed by tanh
        num_half_net_layers = (self._layers_num - 1) // 2

        for layer_num in range(self._layers_num - 1):

            out_layers = self._layers_width
            in_layers = self._RGB_CHANNELS if layer_num == 0 else self._layers_width
            # if layer_num >= num_half_net_layers:
            #     in_layers *= 2

            # print((num_half_net_layers - i % num_half_net_layers) - 1)
            conv = torch.nn.Conv2d(
                in_channels=in_layers,
                out_channels=out_layers,
                padding=self._PADDING,
                kernel_size=self._KERNEL_SHAPE,
                stride=self._STRIDE,
                # device='cuda' if torch.cuda.is_available() else 'cpu',
            )

            # if layer_num >= num_half_net_layers:
            #     layer_skipped_from_unm = (num_half_net_layers - layer_num % num_half_net_layers) - 1
            #     skipped_from_layer = layers[layer_skipped_from_unm]
            #
            #     conv = torch.concatenate((skipped_from_layer, conv))
            #
            #     print(f'{layer_num} to {skipped_from_layer}')
            # else:
            #     print(layer_num)

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
                x += mid_results[layer_num_skip]

        # Last layer that produces the curve maps.
        x = self._layers[-1](x)

        # @@@@@@@@@@@@@@@@ Iterations @@@@@@@@@@@@@@@@@ #
        le = input_image

        # import matplotlib.pyplot as plt

        for i in range(self._iterations_num):

            # RGB_ALPHA_MAP is = (iteration_num + iteration_num + iteration_num, H, W)
            alpha_i = x[:, (i, i + self._iterations_num, i + 2 * self._iterations_num), :, :]
            le = self._light_enhancement_curve_function(prev_le=le, curr_alpha=alpha_i)

            # plt.title(f'Iteration: 8{i + 1}')
            # plt.imshow(le.to('cpu').detach().numpy()[0].transpose(1, 2, 0))
            # plt.show()

        return le

# if __name__ == '__main__':
#
#     config = {
#         ZeroDCE.INPUT_SIZE: 256,
#         # TODO At moment, to ease implementation even number of layers, is supported
#         #  (LAYERS_NUM - 1 is the number of layers in the DCE-NET. Last layer are the curve maps).
#         ZeroDCE.LAYERS_NUM: 7,
#         ZeroDCE.LAYERS_WIDTH: 32,
#         ZeroDCE.ITERATIONS_NUM: 8
#     }
#
#     zero_dce = ZeroDCE(config=config)
#     print(zero_dce)
#
#     # ~~~ Testing stuff ~~~
#     test_image = Image.open('../img.png').convert('RGB')
#
#     transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Resize([256, 256]),
#         # transforms.CenterCrop(224),
#         # transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
#     ])
#     test_input = transform(test_image)
#     test_input = test_input[None, :, :, :]
#
#     zero_dce(x=test_input)
