from data_manager.entity_classes.tne_sample import TNESample
from torch.utils.data import Dataset
from typing import List


# Provides an interface to access samples from the dataset.
class DataAccess(Dataset):
    def __init__(self, data: List[TNESample]) -> None:
        """
            DESCRIPTION: init crucial information used by Data Access object.
            ARGUMENTS:
              - data (LIST[TNESample]): list of the samples in the dataset.
        """
        self.data = data

    def __getitem__(self, index: int) -> TNESample:
        """
          DESCRIPTION: Getter method to get the i-th sample in the dataset given index i
          The method allows to access the data and load the data from.
        """
        return self.data[index]

    def __len__(self) -> int:
        """
          DESCRIPTION: returns the length of the dataset.
          i.e the number of samples/documents the data is composed of.
          RETURN (int): number of samples in dataset
        """
        return len(self.data)
