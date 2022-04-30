from typing import Dict, Optional, Any
import torch
import torch.nn as tnn
from model.enums.activation_function import get_activation_module


class Multilayer_Classifier(tnn.Module):
    def __init__(self, mlp_config: Dict[str, Any], output_probs: bool, device_type: Optional[str] = "cuda") -> None:
        """
            DESCRIPTION: the method init an mlp module according to the given configuration.
            Mlp module is multi-layer feedforward artificial neural network that generates
            a set of outputs from a set of inputs.
            Mlp module is multiple linear fully-connected layers one after the other.
            which can include dropout, activation functions, normalization techniques etc.
            ARGUMENTS:
              - mlp_config (Dict[str, Any]): the configuration of the desirable model.
                The configuration is a map in the following format:
                "num_layers": (int) Number of layers of the model
                "layers_dims": (List[int]) Number of neurons for each layer in the model
                "activation_function":  the desirable activation function (such as relu, tanh)
                "dropout_rate": (float) the probability of each neuron to be dropped out of
                forward pass and back-propagation.
                "batch_norm": (boolean) Indicate whether to use batch normalization or not.
              - device_type (Optional[str]): device type (cuda / cpu)
        """
        super(Multilayer_Classifier, self).__init__()
        self.device_type = device_type
        self.output_probs = output_probs

        ######################################
        #     Training hyper-parameters      #
        ######################################
        self.dropout_rate = mlp_config['dropout_rate']
        self.batch_norm = mlp_config['batch_norm']
        self.activation = mlp_config['activation_function']

        ######################################
        #       Init layers dimensions       #
        ######################################
        self.layers_dims = mlp_config['layers_dims']
        self.nof_layers = mlp_config['num_layers']

        ######################################
        #          Init mlp layers           #
        ######################################
        self.init_layers()

    def init_layers(self) -> None:
        """
            DESCRIPTION: the method init the layers/modules of the model according to
            the given configuration.
            such as linear fully connected layers, activation function, dropout etc.
        """
        # Init linear fully connected layers
        self.linear_layers = []
        for i in range(0, self.nof_layers):
            linear_layer = tnn.Linear(self.layers_dims[i], self.layers_dims[i + 1], bias=True, device=self.device_type)
            torch.nn.init.xavier_uniform_(linear_layer.weight)
            self.linear_layers.append(linear_layer)

        # Init batch normalization
        if self.batch_norm:
            self.bc_layer = tnn.BatchNorm1d(num_features=self.hidden_size)
        else:
            self.bc_layer = lambda x: x

        # Init Softmax
        if self.output_probs:
            self.softmax = tnn.LogSoftmax(dim=-1)
        else:
            self.softmax = lambda x: x

        # Init activation function
        self.activation_function = get_activation_module(self.activation)()

        # Init drop-out
        self.dropout = tnn.Dropout(self.dropout_rate)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        DESCRIPTION: Forward pass
        The calculation process which used to get probabilities from inputs data.
        The data is traversing through all neurons from the first to last layer.
        RETURN (torch.Tensor): encoding of the input data
        """
        # Input for the mlp model
        h = input

        # Feed the output of the previous layer to the current layer,
        # then apply non-linear activation function and dropout.
        for i in range(self.nof_layers - 1):
            z = self.linear_layers[i](h)
            h = self.dropout(self.activation_function(z))

        # Output layer
        z = self.linear_layers[self.nof_layers - 1](h)
        output = self.softmax(z)
        return output
