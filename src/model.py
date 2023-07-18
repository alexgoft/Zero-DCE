import torch


class ZeroDCE(torch.nn.Module):

    LAYERS_NUM = 'layers_num'
    LAYERS_WIDTH = 'layers_width'
    ITERATIONS_NUM = 'iterations_num'
    INPUT_SIZE = '255'  # Input layer have input shape of (INPUT_SIZE, INPUT_SIZE, 3)

    def __init__(self, config):
        super(ZeroDCE, self).__init__()

        self._input_shape = config[self.INPUT_SIZE]
        self._layers_num = config[self.LAYERS_NUM]
        self._layers_width = config[self.LAYERS_WIDTH]
        self._iterations_num = config[self.ITERATIONS_NUM]

        self._layers = []

        # @@@ DCE-Net
        # Every layer is followed by RELU, and the last is followed by tanh
        for i in range(self._layers_num):

            conv = None
            activation = None

            self._layers.append(conv)
            self._layers.append(activation)

    def forward(self, x):

        # DCE-Net feedforward.
        for layer in self._layers:
            x = layer(x)

        # Calculating output via curve-maps
        pass

        return x


if __name__ == '__main__':

    config = {
        ZeroDCE.INPUT_SIZE: 256,
        ZeroDCE.LAYERS_NUM: 8,
        ZeroDCE.LAYERS_WIDTH: 32,
        ZeroDCE.ITERATIONS_NUM: 8
    }

    zero_dce = ZeroDCE(config=config)

    print('The model:')
    print(zero_dce)

    print('\n\nModel params:')
    for param in zero_dce.parameters():
        print(param)
