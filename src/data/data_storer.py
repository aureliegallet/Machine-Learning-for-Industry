import pandas as pd
import copy
import torch


class DataStorer:
    def __init__(self) -> None:
        self._data = None
        self._missing_values_first_index = None
        self._train_x = None
        self._train_y = None
        self._test_x = None
        self._test_y = None
        self._val_x = None
        self._val_y = None
    
    @property
    def missing_values_first_index(self) -> int:
        if self._missing_values_first_index is None:
            raise NameError("Missing values first index is not defined")
        return self._missing_values_first_index

    @missing_values_first_index.setter
    def missing_values_first_index(self, missing_values_first_index: int) -> None:
        if not isinstance(missing_values_first_index, int):
            raise TypeError(f"Missing values first index is not int, got {type(missing_values_first_index)}")
        self._missing_values_first_index = missing_values_first_index

    @property
    def train_x(self) -> pd.DataFrame:
        if self._train_x is None:
            raise NameError("Data was not split yet, unable to load data")
        return copy.deepcopy(self._train_x)

    @train_x.setter
    def train_x(self, train_x: pd.DataFrame) -> None:
        if not isinstance(train_x, pd.DataFrame):
            raise TypeError(f"Train_x is not pandas DataFrame, got {type(train_x)}")
        self._train_x = train_x

    @property
    def train_y(self) -> pd.DataFrame:
        if self._train_y is None:
            raise NameError("Data was not split yet, unable to load data")
        return copy.deepcopy(self._train_y)

    @train_y.setter
    def train_y(self, train_y: pd.DataFrame) -> None:
        if not isinstance(train_y, pd.DataFrame):
            raise TypeError(f"train_y is not pandas DataFrame, got {type(train_y)}")
        self._train_y = train_y

    @property
    def test_x(self) -> pd.DataFrame:
        if self._test_x is None:
            raise NameError("Data was not split yet, unable to load data")
        return copy.deepcopy(self._test_x)

    @test_x.setter
    def test_x(self, test_x: pd.DataFrame) -> None:
        if not isinstance(test_x, pd.DataFrame):
            raise TypeError(f"test_x is not pandas DataFrame, got {type(test_x)}")
        self._test_x = test_x

    @property
    def test_y(self) -> pd.DataFrame:
        if self._test_y is None:
            raise NameError("Data was not split yet, unable to load data")
        return copy.deepcopy(self._test_y)

    @test_y.setter
    def test_y(self, test_y: pd.DataFrame) -> None:
        if not isinstance(test_y, pd.DataFrame):
            raise TypeError(f"test_y is not pandas DataFrame, got {type(test_y)}")
        self._test_y = test_y

    @property
    def val_x(self) -> pd.DataFrame:
        if self._val_x is None:
            raise NameError("Data was not split yet, unable to load data")
        return copy.deepcopy(self._val_x)

    @val_x.setter
    def val_x(self, val_x: pd.DataFrame) -> None:
        if not isinstance(val_x, pd.DataFrame):
            raise TypeError(f"val_x is not pandas DataFrame, got {type(val_x)}")
        self._val_x = val_x

    @property
    def val_y(self) -> pd.DataFrame:
        if self._val_y is None:
            raise NameError("Data was not split yet, unable to load data")
        return copy.deepcopy(self._val_y)

    @val_y.setter
    def val_y(self, val_y: pd.DataFrame) -> None:
        if not isinstance(val_y, pd.DataFrame):
            raise TypeError(f"val_y is not pandas DataFrame, got {type(val_y)}")
        self._val_y = val_y

    @property
    def data(self) -> pd.DataFrame:
        return copy.deepcopy(self._data)

    @data.setter
    def data(self, data: pd.DataFrame) -> None:
        if not isinstance(data, pd.DataFrame):
            raise TypeError(f"data is not pandas DataFrame, got {type(data)}")
        self._data = data

    def pandas_to_tensor(self, data: pd.DataFrame) -> torch.Tensor:
        # Convert pandas dataframe to tensor
        return torch.tensor(data.values)
