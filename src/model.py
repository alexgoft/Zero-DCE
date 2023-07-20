import torch

import torchvision.transforms as transforms
import matplotlib.pyplot as plt

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

        self._input_shape = config[self.INPUT_SIZE]
        self._layers_num = config[self.LAYERS_NUM]
        self._layers_width = config[self.LAYERS_WIDTH]
        self._iterations_num = config[self.ITERATIONS_NUM]

        # @@@ DCE-Net
        #   TODO Below to a init function?
        self._layers = []

        relu = torch.nn.ReLU()
        tanh = torch.nn.Tanh()

        # Every layer is followed by RELU, and the last is followed by tanh
        for i in range(self._layers_num):

            conv = torch.nn.Conv2d(
                in_channels=self._RGB_CHANNELS if i == 0 else self._layers_width,
                out_channels=self._layers_width,
                padding=self._PADDING,
                kernel_size=self._KERNEL_SHAPE,
                stride=self._STRIDE
            )

            self._layers.append(conv)
            self._layers.append(relu)

        # Add final layer that produce the Curve maps
        self._layers.append(torch.nn.Conv2d(
                in_channels=self._layers_width, out_channels=self._iterations_num * self._RGB_CHANNELS,
                padding=self._PADDING, kernel_size=self._KERNEL_SHAPE, stride=self._STRIDE
        ))
        self._layers.append(tanh)

        self._model = torch.nn.Sequential(*self._layers)

    def forward(self, x):

        input_image = x.detach().clone()

        # plt.imshow(input_image.permute(1, 2, 0))
        # plt.show()

        # @@@@@@@@@@@@@@@@@@ DCE-Net @@@@@@@@@@@@@@@@@@ #
        # Create Curve maps for input image x.
        mid_results = []

        # First num_half_net_layers layers. Results will be connected to further layers.
        num_half_net_layers = (self._layers_num - 1) // 2
        for layer_num in range(num_half_net_layers):
            x = self._layers[layer_num](x)

            mid_results.append(x)

        # Skip connections layers.
        for layer_num in range(num_half_net_layers, self._layers_num - 1):
            x = self._layers[layer_num](x) + mid_results[layer_num % num_half_net_layers]

        # Last layer that produces the curve maps.
        x = self._layers[self._layers_num - 1](x)
        print(x.shape)

        # @@@@@@@@@@@@@@@@ Iterations @@@@@@@@@@@@@@@@@ #
        # TODO Implement
        for i in range(self._iterations_num):
            print(f'Iteration #: {i + 1}')

            iteration_i_map = x[:, (i, i + self._iterations_num, i + 2 * self._iterations_num), :, :]

        return x


if __name__ == '__main__':
    config = {
        ZeroDCE.INPUT_SIZE: 256,
        # TODO At moment, to ease implementation even number of layers, is supported
        #  (LAYERS_NUM - 1 is the number of layers in the DCE-NET. Last layer are the curve maps).
        ZeroDCE.LAYERS_NUM: 7,
        ZeroDCE.LAYERS_WIDTH: 32,
        ZeroDCE.ITERATIONS_NUM: 8
    }

    zero_dce = ZeroDCE(config=config)
    print(zero_dce)

    # ~~~ Testing stuff ~~~
    test_image = Image.open('../img.png').convert('RGB')

    # plt.imshow(test_image)
    # plt.show()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize([256, 256]),
        # transforms.CenterCrop(224),
        # transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
    ])
    test_input = transform(test_image)
    test_input = test_input[None, :, :, :]

    out = zero_dce(x=test_input)
    # print(out)
