from enum import Enum, unique
import torch

# Enum type which represent an optimizer
@unique
class Optimizer(Enum):
    Adam = 1
    SGD = 2
    Adagrad = 3
    RMSprop = 4
    AdamW = 5


# Map optimizer enum to the corresponding optimizer module
optimizer_module = {
    Optimizer.Adam: torch.optim.Adam,
    Optimizer.SGD: torch.optim.SGD,
    Optimizer.Adagrad: torch.optim.Adagrad,
    Optimizer.RMSprop: torch.optim.RMSprop,
    Optimizer.AdamW: torch.optim.AdamW
}


def get_optimizer_module(optimizer: Optimizer) -> torch.optim:
    """
    DESCRIPTION: the method receives type of optimizer as enum,
    and returns the corresponding optimizer module.
    ARGUMENTS:
      - optimizer (Optimizer): type of desirable optimizer
    """
    return optimizer_module[optimizer]
