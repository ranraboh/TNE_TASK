from enum import Enum, unique
import torch


# Enum type which represent loss function
@unique
class Loss_Function(Enum):
    Cross_Entropy = 1


# Map loss type enum to the corresponding loss module
loss_modules = {
    Loss_Function.Cross_Entropy: torch.nn.CrossEntropyLoss
}


def get_loss_module(loss_type: Loss_Function) -> torch.nn:
    """
      DESCRIPTION: the method receives type of optimizer as enum,
      and returns the corresponding optimizer module.
      ARGUMENTS:
        - optimizer (Optimizer): type of desirable optimizer
    """
    return loss_modules[loss_type]
