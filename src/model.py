import torch

import torchvision.transforms as transforms
import numpy as np


class ZeroDCE(torch.nn.Module):
    LAYERS_NUM = 'layers_num'
    LAYERS_WIDTH = 'layers_width'
    ITERATIONS_NUM = 'iterations_num'
    INPUT_SIZE = 'input_size'  # Input layer have input shape of (INPUT_SIZE, INPUT_SIZE, 3)

    _RGB_CHANNELS = 3

    def __init__(self, config):
        super(ZeroDCE, self).__init__()

        self._input_shape = config[self.INPUT_SIZE]
        self._layers_num = config[self.LAYERS_NUM]
        self._layers_width = config[self.LAYERS_WIDTH]
        self._iterations_num = config[self.ITERATIONS_NUM]

        # TODO Below to a init function?
        # @@@ DCE-Net
        self._layers = []

        # Every layer is followed by RELU, and the last is followed by tanh
        for i in range(self._layers_num):
            in_channels = self._RGB_CHANNELS if i == 0 else self._layers_width
            out_channels = self._iterations_num * self._RGB_CHANNELS if i == self._layers_num - 1 else self._layers_width

            conv = torch.nn.Conv2d(
                padding=1,
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(3, 3),
                stride=(1, 1)
            )
            # Append convolution layer and act function (Tanh for last layer and ReLU for the rest.
            self._layers.append(conv)
            self._layers.append(torch.nn.ReLU() if i != self._layers_num - 1 else torch.nn.Tanh())

        self._model = torch.nn.Sequential(*self._layers)

    def forward(self, x):

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
    # transform = transforms.Compose([
    #     # transforms.Resize(256),
    #     # transforms.CenterCrop(224),
    #     transforms.ToTensor(),
    #     # transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
    # ])
    # dummy_input = transform(np.ones(shape=(256, 256, 3), dtype=np.float32))

    dummy_input = torch.randn(1, 3, 256, 256)
    out = zero_dce(x=dummy_input)
    # print(out)

    # # ------------------------------ #
    # # TODO Debugging visualizations  #
    # # ------------------------------ #
    # import torchvision
    # from torchview import draw_graph
    #
    # model = ZeroDCE(config=config)
    # # if torch.cuda.is_available():
    # #     model.to('cuda')
    # model_graph = draw_graph(model, input_size=(1, 3, 256, 256), expand_nested=True)
    # # model_graph.visual_graph
