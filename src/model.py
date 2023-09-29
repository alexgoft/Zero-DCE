import torch


class ZeroDCE(torch.nn.Module):
    LAYERS_NUM = 'layers_num'
    LAYERS_WIDTH = 'layers_width'
    ITERATIONS_NUM = 'iterations_num'
    INPUT_SIZE = 'input_size'  # Input layer have input shape of (INPUT_SIZE, INPUT_SIZE, 3)
    MODEL_PATH = 'model_path'

    _RGB_CHANNELS = 3

    _DNN_STRIDE = (1, 1)
    _DNN_KERNEL_SHAPE = (3, 3)
    _DNN_CONV_PAD = 1  # use padding 1 to keep same shape between convolutions.

    def __init__(self, config, device, model_path=None):
        super(ZeroDCE, self).__init__()
        self._device = device
        self._model_path = model_path

        # Framework params
        self._input_size = config.model.input_size
        self._layers_num = config.model.layers_num
        self._layers_width = config.model.layers_width
        self._iterations_num = config.model.iterations_num

        # Activation functions
        self._relu = torch.nn.ReLU()
        self._tanh = torch.nn.Tanh()

        # Layers initialization
        self._layers = self._initialize_dce_net_layers()
        self._model = torch.nn.Sequential(*self._layers)

        self._init_weights()

        if self._model_path is None:
            print('[INFO] No model path was given. Creating new model...')
            self._model.train()

        else:
            print(f'[INFO] Loading model from path: {model_path}')
            state_dict = torch.load(model_path)

            # TODO Patch. Remove for model of new versions
            #  (after trained with fix).
            for key in list(state_dict.keys()):
                state_dict[key.replace('_model.', '')] = state_dict.pop(key)

            self._model.load_state_dict(state_dict)
            self._model.eval()

        self._model.to(device=device)
        print(self._model)

    def _init_weights(self):
        """
        The filter weights of each layer are initialized with
        standard zero mean and 0.02 standard deviation Gaussian function.
        Bias is initialized as a constant.
        """
        for layer in self._model:  # Assume all layers are Conv2d.
            torch.nn.init.normal_(layer.weight, mean=0.0, std=0.02)
            layer.bias.data.fill_(0)

    def _initialize_dce_net_layers(self):

        layers = []

        # Every layer is followed by RELU, and the last is followed by tanh
        for layer_num in range(self._layers_num):
            in_layers = self._RGB_CHANNELS if layer_num == 0 else self._layers_width
            out_layers = self._layers_width if layer_num != self._layers_num - 1 else self._iterations_num * self._RGB_CHANNELS

            # print((num_half_net_layers - i % num_half_net_layers) - 1)
            conv = torch.nn.Conv2d(
                in_channels=in_layers,
                out_channels=out_layers,
                padding=self._DNN_CONV_PAD,
                kernel_size=self._DNN_KERNEL_SHAPE,
                stride=self._DNN_STRIDE,
                device=self._device,
            )
            layers.append(conv)

        return layers

    @staticmethod
    def _light_enhancement_curve_function(prev_le, curr_alpha):
        """
        This function represent a Higher-Order Curve:

        LEn(x) = LEn−1(x) + αnLEn−1(x)(1 − LEn−1(x)) s.t LEn−1(x) I(x) when n=1

            LE(I(x); α) is the enhanced version of the given input I(x),
            α ∈ [−1, 1] is the trainable curve parameter
        """
        curr_le = prev_le + curr_alpha * (prev_le - torch.square(prev_le))

        return curr_le

    def forward(self, x):

        input_image = x.detach().clone()

        # @@@@@@@@@@@@@@@@@@ DCE-Net @@@@@@@@@@@@@@@@@@ #
        # Create Curve maps for input image x.
        mid_results = []

        # First num_half_net_layers layers. Results will be connected to further layers.
        for layer_num in range(self._layers_num - 1):
            if layer_num < self._layers_num // 2:  # First half of the net.
                x = self._relu(self._layers[layer_num](x))
                mid_results.append(x)

                # print('conv2d:', layer_num)

            else:  # Skip connections from the first half of the net.
                skipped_layer = (len(self._layers) - 2) - layer_num
                x = self._relu(x + mid_results[skipped_layer])  # x here is the result of the previous layer.

                # print(f'conv2d + skip: {layer_num} + {skipped_layer} [input for layer {layer_num+1}]')

        # Last layer that produces the curve maps
        x = self._tanh(self._layers[-1](x))

        # @@@@@@@@@@@@@@@@ Iterations @@@@@@@@@@@@@@@@@ #
        # Split the curve maps into alpha maps (3 maps for each iteration)
        alpha_maps = torch.split(x, split_size_or_sections=3, dim=1)
        le = input_image

        for i, alpha_i in enumerate(alpha_maps):
            le = self._light_enhancement_curve_function(prev_le=le, curr_alpha=alpha_i)

        return le, torch.concat(alpha_maps, dim=1)  # We need maps for the Illumination Smoothness Loss
