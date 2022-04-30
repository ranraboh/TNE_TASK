from enum import Enum, unique
import torch


# Enum type which represent activation function
@unique
class Activation_Function(Enum):
    ReLU = 1
    TANH = 2
    PReLU = 3
    LeakyReLU = 4


# Map activation function enum to the corresponding activation layer
activation_module = {
    Activation_Function.ReLU: torch.nn.ReLU,
    Activation_Function.TANH: torch.nn.Tanh,
    Activation_Function.PReLU: torch.nn.PReLU,
    Activation_Function.LeakyReLU: torch.nn.LeakyReLU
}


def get_activation_module(activation_type: Activation_Function):
    """
        DESCRIPTION: the method receives type of activation function as enum (such as relu, tanh, prelu etc)
        and returns the corresponding activation module.
        ARGUMENTS:
          - activation_type (Activation_Function): type of desirable activation function
    """
    return activation_module[activation_type]
