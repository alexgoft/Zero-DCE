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
        self._convs = []
        self._acts = []
        # Every layer is followed by RELU, and the last is followed by tanh
        for i in range(self._layers_num):
            in_channels = self._RGB_CHANNELS if i == 0 else self._layers_width
            out_channels = self._iterations_num if i == self._layers_num - 1 else self._layers_width
            conv = torch.nn.Conv2d(
                padding=1,
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(3, 3),
                stride=(1, 1)
            )
            activation = torch.nn.ReLU() if i != self._layers_num - 1 else torch.nn.Tanh()

            self._convs.append(conv)
            self._acts.append(activation)

    def forward(self, x):

        # DCE-Net feedforward.
        mid_results = []
        for layer_num, (conv, act) in enumerate(zip(self._convs, self._acts)):

            print('curr_layer:', layer_num)

            possible_layer_num_to_skip = self._layers_num - layer_num - 1
            if layer_num > possible_layer_num_to_skip:
                x += mid_results[possible_layer_num_to_skip]

                print(f'\t connect {possible_layer_num_to_skip} to {layer_num}')

            x = conv(x)
            x = act(x)

            # TODO 1 This is not elegant enough
            # TODO 2 What about output shape which needs to be (batch_num, num_iterations, 3, 256, 256)
            if layer_num < possible_layer_num_to_skip:
                mid_results.append(x)

            # print(x.shape)

        # Calculating output via curve-maps
        pass

        return x


if __name__ == '__main__':
    config = {
        ZeroDCE.INPUT_SIZE: 256,
        ZeroDCE.LAYERS_NUM: 7,
        ZeroDCE.LAYERS_WIDTH: 32,
        ZeroDCE.ITERATIONS_NUM: 8
    }

    zero_dce = ZeroDCE(config=config)

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