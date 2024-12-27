import io
from typing import Any, Union
import zipfile
import requests
import re
from datetime import datetime, timedelta
import pandas as pd
import os
from scipy import stats
import numpy as np
import pickle as pk
import torch


class DataProcessor:
    def __call__(self, time_frame: int, *args: Any, **kwds: Any) \
            -> tuple[Union[torch.Tensor, None], list[str], list[str]]:
        """Pipeline for model deployment

        Args: time_frame (int): number of days to feed the model

        Returns:
            tuple[torch.Tensor, list[str], list[str]]: torch loader for NN models, errors, warnings
        """
        nn_loader = None

        # Retrieve data and check if the process results in errors
        pollution_data, errors, warnings = self.get_past_pollution_data(time_frame, False)
        if errors:
            return nn_loader, errors, warnings

        climate_data = self.get_climate_data(time_frame)
        train_data = pd.read_csv(os.path.join(os.getcwd(), r"../data/processed", f"train_x.csv"))

        # Check if the retrieved data has problems
        new_errors, new_warnings = self.problems_in_dataset(climate_data, train_data)
        errors += new_errors
        warnings += new_warnings

        if errors:
            return nn_loader, errors, warnings

        preprocessed_climate_data = self.preprocess_climate_data(climate_data, train_data)

        feature_engineered_data = self.feature_engineering(preprocessed_climate_data)
        nn_loader = self.get_nn_loader(feature_engineered_data, pollution_data)

        return nn_loader, errors, warnings

    def get_climate_data(self, time_frame: int) -> pd.DataFrame:
        starting_date = (datetime.today() - timedelta(days=time_frame)).strftime('%Y%m%d')
        end_date = (datetime.today() - timedelta(days=1)).strftime('%Y%m%d')
        lines_to_keep = []

        # Load up-to-date climate data
        climate_data_url = "https://cdn.knmi.nl/knmi/map/page/klimatologie/gegevens/daggegevens/etmgeg_260.zip"
        response = requests.get(climate_data_url, stream=True)
        with zipfile.ZipFile(io.BytesIO(response.content)).open('etmgeg_260.txt') as file:
            reading_data = False

            for line in file.readlines():
                line = line.decode('utf-8')
                if re.match(r'.*{}.*'.format(starting_date), line):
                    reading_data = True
                if re.match(r'#.*', line) or reading_data:
                    lines_to_keep.append(line.replace(" ", "").strip().split(','))
                if re.match(r'.*{}.*'.format(end_date), line):
                    break

        dataframe = pd.DataFrame(lines_to_keep[1:], columns=lines_to_keep[0]).astype(int)
        return dataframe

    def get_past_pollution_data(self, time_frame: int, include_today: bool) \
            -> tuple[pd.DataFrame, list[str], list[str]]:
        errors = []
        warnings = []
        number_of_pages = int((time_frame + 1) * 24 / 50) + 1
        if include_today:
            start_date = 0
            time_frame -= 1
        else:
            start_date = 1

        allowed_dates = []
        for past_days in range(start_date, time_frame + 1):
            allowed_dates.append((datetime.today() - timedelta(days=past_days)).strftime('%Y-%m-%d'))

        pollution_dataframe = pd.DataFrame(columns=["YYYYMMDD", "Average_no2", "Average_o3"])

        for pollutant in ["NO2", "O3"]:
            data = []

            # Download past pollution data from the api
            for station_id in "NL10636", "NL10639", "NL10641", "NL10643":
                full_path = f"https://api.luchtmeetnet.nl/open_api/stations/{station_id}/measurements"
                for page in range(number_of_pages):
                    response = requests.get(f"{full_path}?formula={pollutant}&pages={page}")
                    if response.status_code == 200:
                        for datapoint in response.json()["data"]:
                            if datapoint["timestamp_measured"][:10] in allowed_dates:
                                data.append({
                                    "station_id": station_id,
                                    "value": datapoint["value"],
                                    'timestamp': pd.to_datetime(datapoint['timestamp_measured'])
                                })
                    else:
                        warnings.append(
                            f"It was not possible to retrieve (part of) past {pollutant} data "
                            f"of the station number {station_id} (request to page {page}).")

            if len(data) == 0:
                errors.append(f"It was not possible to retrieve past {pollutant} data.")
                break

            dataframe = pd.DataFrame(data)
            dataframe['date'] = dataframe['timestamp'].dt.strftime('%Y%m%d')

            daily_average_per_station = dataframe.groupby(['station_id', 'date'])['value'].mean().reset_index()
            daily_average = daily_average_per_station.groupby('date')['value'].mean().reset_index()

            if set(pd.to_datetime(daily_average["date"], format='%Y%m%d')
                    .dt.strftime('%Y-%m-%d')) != set(allowed_dates):
                errors.append("There is no past pollution data for every day of the selected time frame")
                break

            pollution_dataframe["YYYYMMDD"] = daily_average["date"]
            if pollutant == "NO2":
                pollution_dataframe["Average_no2"] = daily_average["value"]
            else:
                pollution_dataframe["Average_o3"] = daily_average["value"]

        return pollution_dataframe, errors, warnings

    def problems_in_dataset(self, climate_data: pd.DataFrame, train_data: pd.DataFrame) -> tuple[list[str], list[str]]:
        # Implement a data monitoring and alert mechanism for relevant events like missing data, data
        # distribution shift, and out-of-training min-max feature values

        errors = []
        warnings = []

        for column in train_data.columns:
            # Error if a column is missing
            if column not in climate_data.columns:
                errors.append(f"The {column} feature is missing.")
            # Error if data is missing
            elif climate_data[column].isna().any():
                errors.append(f"The {column} has missing values.")
            # Only analyse distributions and range of meaningful columns
            elif column != "YYYYMMDD":
                train_min = train_data[column].min()
                train_max = train_data[column].max()
                data_min = climate_data[column].min()
                data_max = climate_data[column].max()

                # Warning if out of min-max range
                if data_min < train_min or data_max > train_max:
                    warnings.append(f"The values of the {column} feature are out-of-training min-max.")

                # Warning if new datapoints shift the original distribution
                _, p_value = stats.ks_2samp(train_data[column], pd.concat([train_data[column], climate_data[column]]))
                if p_value < 0.05:
                    warnings.append(
                        f"The values of the {column} shift the original distribution. "
                        f"Test = two-sample Kolmogorov-Smirnov test, p={p_value:.2f}.")

        return errors, warnings

    def preprocess_climate_data(self, climate_data: pd.DataFrame, train_data: pd.DataFrame) -> pd.DataFrame:
        preprocessed_climate_data = climate_data.drop(columns="#STN")

        # Normalise the dataset using robust scaler
        scaler = pk.load(open("../src/data/scaler.pkl", "rb"))
        # The date cannot be normalised
        columns = preprocessed_climate_data.columns.drop("YYYYMMDD")
        preprocessed_climate_data[columns] = scaler.transform(climate_data[columns])

        return preprocessed_climate_data

    def feature_engineering(self, climate_data: pd.DataFrame) -> pd.DataFrame:
        # Extract Seasonal Information
        climate_data = self._transform_time(climate_data, "YYYYMMDD", 366, "YYYYMMDD")
        climate_data = climate_data.drop(columns="YYYYMMDD")

        # Apply PCA to dataset
        pca_object = pk.load(open("../results/best_model/transformation_object.pkl", "rb"))
        climate_data = pd.DataFrame(data=pca_object.transform(climate_data),
                                    columns=[f"Component_{i + 1}" for i in range(pca_object.n_components_)])

        return climate_data

    def _transform_time(self, data: pd.DataFrame, col_name: str, max_value: int, time_format: str) -> pd.DataFrame:
        """Transforms numerical days (0-364) into a cyclic representation (sin, cos)

        Args:
            data (pd.DataFrame): The pandas DataFrame from which the days variable needs to be transformed.
            Note that the time column needs to be in the format: YYYYMMDD!!
            col_name (str): The name of the column which holds the time data.
            max_value (int): The maximal value occurring in the time column.
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

    def get_nn_loader(self, climate_data: pd.DataFrame, pollution_data: pd.DataFrame) -> torch.Tensor:
        full_data = pd.concat([climate_data, pollution_data[["Average_no2", "Average_o3"]]], axis=1)

        return torch.tensor([full_data.to_numpy()], dtype=torch.float64)
