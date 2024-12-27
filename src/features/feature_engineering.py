import pandas as pd
import pickle as pk
import numpy as np
import copy
import os
import umap
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from src.data.data_storer import DataStorer

root_path = os.getcwd()


class FeatureEngineering:
    def __init__(self, days_to_lag: int, pca_in_pipeline: bool = False) -> None:
        # Assert that days_to_lag is an integer
        assert isinstance(days_to_lag, int)
        self._days_to_lag = days_to_lag
        self.pca_in_pipeline = pca_in_pipeline

    def __call__(self, data_storer: DataStorer) -> tuple[DataStorer, list[DataStorer], list[DataStorer]]:
        """Applies a preprocessing pipeline as described in the report

        Args:
            data_storer (DataStorer): The instance of DataStorer storing the model's data.
        """

        # Extract Seasonal Information. No need to drop date anymore because feature extraction happens in hp tuning
        data_storer.train_x = self._transform_time(data_storer.train_x, "YYYYMMDD", 366, "YYYYMMDD")
        data_storer.val_x = self._transform_time(data_storer.val_x, "YYYYMMDD", 366, "YYYYMMDD")
        data_storer.test_x = self._transform_time(data_storer.test_x, "YYYYMMDD", 366, "YYYYMMDD")

        # Apply PCA to train and test dataset
        if self.pca_in_pipeline:
            # Drop dates
            date_train = data_storer.train_x['YYYYMMDD']
            data_storer.train_x = data_storer.train_x.drop(columns="YYYYMMDD")
            date_val = data_storer.val_x['YYYYMMDD']
            data_storer.val_x = data_storer.val_x.drop(columns="YYYYMMDD")
            date_test = data_storer.test_x['YYYYMMDD']
            data_storer.test_x = data_storer.test_x.drop(columns="YYYYMMDD")

            # Apply pca
            pca_object = self._get_pca(pca_data=data_storer.train_x)

            data_storer.train_x = pd.DataFrame(data=pca_object.transform(data_storer.train_x),
                                               columns=[f"Component_{i + 1}" for i in range(pca_object.n_components_)])
            data_storer.test_x = pd.DataFrame(data=pca_object.transform(data_storer.test_x),
                                              columns=[f"Component_{i + 1}" for i in range(pca_object.n_components_)])
            data_storer.val_x = pd.DataFrame(data=pca_object.transform(data_storer.val_x),
                                             columns=[f"Component_{i + 1}" for i in range(pca_object.n_components_)])
            
            # Re add dates
            data_storer.train_x = pd.concat(
                [data_storer.train_x.reset_index(drop=True), date_train.reset_index(drop=True)],
                axis=1, join="inner"
            )
            data_storer.val_x = pd.concat(
                [data_storer.val_x.reset_index(drop=True), date_val.reset_index(drop=True)],
                axis=1, join="inner"
            )
            data_storer.test_x = pd.concat(
                [data_storer.test_x.reset_index(drop=True), date_test.reset_index(drop=True)],
                axis=1, join="inner"
            )

        # Transform date column into an array of numbers from 0 to length of the dataset
        date_format_train = copy.deepcopy(data_storer.train_x)
        date_format_train['YYYYMMDD'] = pd.to_datetime(data_storer.train_x['YYYYMMDD'], format='%Y%m%d')
        start_date = pd.to_datetime("20100101", format='%Y%m%d')
        date_format_train['YYYYMMDD'] = (date_format_train['YYYYMMDD'] - start_date).dt.days
        data_storer.train_x = date_format_train

        date_format_val = copy.deepcopy(data_storer.val_x)
        date_format_val['YYYYMMDD'] = pd.to_datetime(data_storer.val_x['YYYYMMDD'], format='%Y%m%d')
        start_date = pd.to_datetime("20100101", format='%Y%m%d')
        date_format_val['YYYYMMDD'] = (date_format_val['YYYYMMDD'] - start_date).dt.days
        data_storer.val_x = date_format_val

        date_format_test = copy.deepcopy(data_storer.test_x)
        date_format_test['YYYYMMDD'] = pd.to_datetime(data_storer.test_x['YYYYMMDD'], format='%Y%m%d')
        start_date = pd.to_datetime("20100101", format='%Y%m%d')
        date_format_test['YYYYMMDD'] = (date_format_test['YYYYMMDD'] - start_date).dt.days
        data_storer.test_x = date_format_test

        neural_networks_loader = copy.deepcopy(data_storer)
        regression_loader = self._shift_data(data_storer)
        regression_loader_feature_selected = copy.deepcopy(regression_loader)

        # This code can be used in case one wants to re-explore feature selection using random forests
        # Apply feature selection for regression_loader and SVM:
        # for i, storer in enumerate(regression_loader_feature_selected):
        #     features_list = self._feature_selection_rf(storer.train_x, storer.train_y)

        #     storer.train_x = storer.train_x.iloc[: , features_list]
        #     storer.test_x = storer.test_x.iloc[: , features_list]
        #     storer.val_x = storer.val_x.iloc[: , features_list]

        return neural_networks_loader, regression_loader, regression_loader_feature_selected

    def create_pca(self, data: pd.DataFrame) -> PCA:
        pca = PCA()
        pca.fit_transform(data)
        return pca
    
    def create_umap(self, data: pd.DataFrame, features_out: int) -> umap.UMAP:
        umap_object = umap.UMAP(n_components=features_out)
        umap_object.fit(data)
        return umap_object

    def apply_pca(self, data: pd.DataFrame, is_train: bool = False) -> pd.DataFrame:
        if is_train:
            pca_object = self._get_pca(data)
        else:
            pca_object = self._get_pca()

        data = pd.DataFrame(data=pca_object.transform(data),
                            columns=[f"PCA_component_{i + 1}" for i in range(pca_object.n_components_)])
        return data

    def _get_pca(self, pca_data: pd.DataFrame = None) -> PCA:
        """If you want to create a new pca object, give it the data and it will use that one.
        """
        if pca_data is not None:
            try:
                os.remove((root_path, "src/features/pca.pkl"))
            except Exception:
                pass
            pca = PCA()
            pca.fit_transform(pca_data)
            pk.dump(pca, open(os.path.join(root_path, "src/features/pca.pkl"), "wb"))
            return pca
        pca = pk.load(open(os.path.join(root_path, "src/features/pca.pkl"), "rb"))
        return pca

    def apply_umap(self, data: pd.DataFrame, features_out: int, is_train: bool = False) -> pd.DataFrame:
        if is_train:
            umap_object = self._get_umap(features_out, data)
        else:
            umap_object = self._get_umap(features_out)

        data = pd.DataFrame(data=umap_object.transform(data),
                            columns=[f"UMAP_component_{i + 1}" for i in range(umap_object.n_components)])
        return data

    def _get_umap(self, features_out: int, umap_data: pd.DataFrame = None) -> umap.UMAP:
        if umap_data is not None:
            try:
                os.remove(os.path.join(root_path, "src/features/umap.pkl".format(features_out)))
            except Exception:
                pass
            umap_object = umap.UMAP(n_components=features_out)
            umap_object.fit(umap_data)
            pk.dump(umap_object, open(os.path.join(root_path, "src/features/umap.pkl".format(features_out)), "wb"))
            return umap_object
        umap_object = pk.load(open(os.path.join(root_path, "src/features/umap.pkl".format(features_out)), "rb"))
        return umap_object

    def _feature_selection_rf(self, x_train: DataStorer, y_train: DataStorer, feature_selection_threshold=0.02) -> list:
        """ Apply random forests and evaluate them based on mean decrease in impurity (measure for random forests)
        """
        feature_names = [f"component_{i}" for i in range(41)]
        rf = RandomForestRegressor(random_state=0)
        rf.fit(x_train, y_train)
        importances = rf.feature_importances_
        forest_importances = pd.Series(importances, index=feature_names)
        integer_list = [int(index.split('_')[1]) for index in
                        forest_importances[importances > feature_selection_threshold].index]
        print(f"Feature columns to use as input: {integer_list}")
        return integer_list

    def _shift_data(self, data_storer: DataStorer) -> list[DataStorer]:
        regression_loader = []
        for shift in range(self._days_to_lag):
            regression_loader.append(self._shift_data_day(data_storer, shift + 1))
        return regression_loader

    def _shift_data_day(self, data_storer: DataStorer, day: int) -> DataStorer:
        data_storer_day = DataStorer()
        data_storer_day.train_x = pd.DataFrame(columns=data_storer.train_x.columns)
        data_storer_day.train_y = pd.DataFrame(columns=data_storer.train_y.columns)
        data_storer_day.val_x = pd.DataFrame(columns=data_storer.val_x.columns)
        data_storer_day.val_y = pd.DataFrame(columns=data_storer.val_y.columns)
        data_storer_day.test_x = pd.DataFrame(columns=data_storer.test_x.columns)
        data_storer_day.test_y = pd.DataFrame(columns=data_storer.test_y.columns)

        for i in range(len(data_storer.train_x)-day):
            if data_storer.train_x.iloc[i]["YYYYMMDD"] + day == data_storer.train_x.iloc[i+day]["YYYYMMDD"]:
                data_storer_day.train_x = pd.concat([data_storer_day.train_x, data_storer.train_x.iloc[[i]]],
                                                    ignore_index=True)
                data_storer_day.train_y = pd.concat([data_storer_day.train_y, data_storer.train_y.iloc[[i+day]]],
                                                    ignore_index=True)

        for i in range(len(data_storer.val_x)-day):
            if data_storer.val_x.iloc[i]["YYYYMMDD"] + day == data_storer.val_x.iloc[i+day]["YYYYMMDD"]:
                data_storer_day.val_x = pd.concat([data_storer_day.val_x, data_storer.val_x.iloc[[i]]],
                                                  ignore_index=True)
                data_storer_day.val_y = pd.concat([data_storer_day.val_y, data_storer.val_y.iloc[[i+day]]],
                                                  ignore_index=True)

        for i in range(len(data_storer.test_x)-day):
            if data_storer.test_x.iloc[i]["YYYYMMDD"] + day == data_storer.test_x.iloc[i+day]["YYYYMMDD"]:
                data_storer_day.test_x = pd.concat([data_storer_day.test_x, data_storer.test_x.iloc[[i]]],
                                                   ignore_index=True)
                data_storer_day.test_y = pd.concat([data_storer_day.test_y, data_storer.test_y.iloc[[i+day]]],
                                                   ignore_index=True)

        data_storer_day.train_x = data_storer_day.train_x.drop(columns='YYYYMMDD')
        data_storer_day.val_x = data_storer_day.val_x.drop(columns='YYYYMMDD')
        data_storer_day.test_x = data_storer_day.test_x.drop(columns='YYYYMMDD')

        return data_storer_day

    def _transform_time(self, data: pd.DataFrame, col_name: str, max_value: int, time_format: str) -> pd.DataFrame:
        """Transforms numerical days (0-364) into a cyclic representation (sin, cos)

        Args:
            data (pd.DataFrame): The pandas DataFrame from which the days variable needs to be transformed.
            Note that the time column needs to be in the format: YYYYMMDD!!
            col_name (str): The name of the column which holds the time data.
            max_value (int): The maximal value occuring in the time column.
            If you have a year with 365 days, 365 is the maximum value (then a new cycle begins).

        This method is based on the following tutorial:
        https://www.kaggle.com/code/avanwyk/encoding-cyclical-features-for-deep-learning
        For more:
        https://www.homebound.com/learn/encoding-seasonality-to-improve-our-property-valuation-models
        """
        if not isinstance(col_name, str):
            raise TypeError(f"Expected {str} for argument col_name, got {type(col_name)}")
        if not isinstance(max_value, int):
            raise TypeError(f"Expected {type(1)} for argument max_value, got {type(max_value)}")
        if not isinstance(data, pd.DataFrame):
            raise TypeError(f"Expected pandas DataFrame for argument data, got {type(data)}")

        if time_format == "D":
            # Assumes time column starts at 0 and ends at 364 each cycle.
            data[col_name + '_sin'] = np.sin(2 * np.pi * data[col_name] / max_value)
            data[col_name + '_cos'] = np.cos(2 * np.pi * data[col_name] / max_value)
        elif time_format == "YYYYMMDD":
            temp = pd.DataFrame()
            temp[col_name] = pd.to_datetime(data[col_name].astype(str), format='%Y%m%d')
            temp[col_name] = (temp[col_name].dt.dayofyear - 1)
            data[col_name + '_sin'] = np.sin(2 * np.pi * temp[col_name] / max_value)
            data[col_name + '_cos'] = np.cos(2 * np.pi * temp[col_name] / max_value)
        else:
            raise ValueError("Variable time_format must be either YYYYMMDD or D")
        return data
