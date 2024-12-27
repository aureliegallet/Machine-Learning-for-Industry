import copy
from typing import Any

import pandas as pd
from torch.utils.data import DataLoader as TorchDataLoader, DataLoader
from src.features.NNLoader import NNLoader
from src.data.data_storer import DataStorer


# A dataStorer loader class for hyper parameter tuning
class HPDataLoader:
    def __init__(self, data: DataStorer, days_lagged: int = 2) -> None:
        self._data_stored = data
        self._days_lagged = days_lagged

    @property
    def data_stored(self) -> DataStorer:
        return copy.deepcopy(self._data_stored)
    
    def load_data_loader(self, train_x: pd.DataFrame, train_y: pd.DataFrame, val_x: pd.DataFrame, val_y: pd.DataFrame) \
            -> tuple[DataLoader[Any], DataLoader[Any]]:
        training_data = NNLoader(train_x, train_y, self._days_lagged)
        validation_data = NNLoader(val_x, val_y, self._days_lagged)
        training_data = TorchDataLoader(training_data, batch_size=1, shuffle=False)
        validation_data = TorchDataLoader(validation_data, batch_size=1, shuffle=False)
        return training_data, validation_data
