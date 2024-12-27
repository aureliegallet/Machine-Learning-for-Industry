from typing import Any
import pandas as pd
from pandas import DataFrame
from sklearn.preprocessing import RobustScaler
from src.data.data_storer import DataStorer
import os
import copy
import pickle as pk


class PreProcessor:
    def __call__(self, data_storer: DataStorer, *args: Any, **kwds: Any) -> None:
        """Applies preprocessing as described in the report.

        Args:
            data_storer (DataStorer): The instance of DataStorer storing the model's data.
        """        
        # Pipeline for data preprocessing
        data_storer.data = self._remove_missing_values(data_storer)

        # Split datasets
        (data_storer.train_x, data_storer.train_y, data_storer.val_x, data_storer.val_y, data_storer.test_x,
         data_storer.test_y,) = self._split_dataframe(data_storer.data)

        # Store train dataset
        data_storer.train_x.to_csv(os.path.join(os.getcwd(), r"data/processed", "train_x.csv"), index=False)

        # Normalization
        data_storer.train_x, data_storer.test_x, data_storer.val_x = self._normalize(data_storer.train_x,
                                                                                     data_storer.test_x,
                                                                                     data_storer.val_x)

        # Create new csv
        data_storer.data.to_csv(os.path.join(os.getcwd(), r"data/processed", "processed_data.csv"), index=False)

    def _remove_missing_values(self, data_loader: DataStorer) -> pd.DataFrame:
        # first_na_index = data_loader.data[data_loader.data.isnull().any(axis=1)].index[0]
        # data_loader.missing_values_first_index = int(first_na_index)

        # Don't include rows without values to predict or it will break the plots
        data = data_loader.data.dropna()
        return data

    def _split_dataframe(self, data: pd.DataFrame):
        # Splits the dataframe into train_x and train_y, val_x and val_y, test_x and test_y
        data["YYYYMMDD"] = pd.to_datetime(data["YYYYMMDD"], format="%Y%m%d")

        # Validation is the year 2018
        val = data[data["YYYYMMDD"].dt.year.isin(range(2018, 2019))]

        # Test is December, April, July, and October of 2019 - 2023
        test = copy.deepcopy(data)
        test['YYYYMMDD'] = pd.to_datetime(test['YYYYMMDD'], format='%Y%m%d')
        years = list(range(2019, 2024))  # Years
        months = [4, 7, 10, 12]  # April, July, October, December
        test = test[(test['YYYYMMDD'].dt.year.isin(years)) & (test['YYYYMMDD'].dt.month.isin(months))]

        # Train is all other data
        combined_val_test = pd.concat([val, test], ignore_index=True)
        train = data[~data['YYYYMMDD'].isin(combined_val_test['YYYYMMDD'])]

        train_x = train.drop(columns=["Average_no2", "Average_o3"])
        val_x = val.drop(columns=["Average_no2", "Average_o3"])
        test_x = test.drop(columns=["Average_no2", "Average_o3"])

        train_y = train[["Average_no2", "Average_o3"]]
        val_y = val[["Average_no2", "Average_o3"]]
        test_y = test[["Average_no2", "Average_o3"]]

        train_x["YYYYMMDD"] = train_x["YYYYMMDD"].dt.strftime('%Y%m%d')
        test_x["YYYYMMDD"] = test_x["YYYYMMDD"].dt.strftime('%Y%m%d')
        val_x["YYYYMMDD"] = val_x["YYYYMMDD"].dt.strftime('%Y%m%d')

        return train_x, train_y, val_x, val_y, test_x, test_y

    def _normalize(self, train: pd.DataFrame, test: pd.DataFrame, val: pd.DataFrame) -> \
            tuple[DataFrame, DataFrame, DataFrame]:
        # Normalise the dataset using robust scaler
        scaler = RobustScaler()
        # The date cannot be normalised
        columns = train.columns.drop("YYYYMMDD")
        train[columns] = scaler.fit_transform(train[columns])
        test[columns] = scaler.transform(test[columns])
        val[columns] = scaler.transform(val[columns])
        root_path = os.getcwd()
        pk.dump(scaler, open(os.path.join(root_path, "src/data/scaler.pkl"), "wb"))
        return train, test, val
