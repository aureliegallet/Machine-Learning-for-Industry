import re
import pandas as pd
from pandas import DataFrame
import numpy as np
import os
import copy


class DataFormatter:
    def __init__(self) -> None:
        self._data = None
        self._climate_data = None
        self._no2_pollution_data = None
        self._o3_pollution_data = None

    @property
    def data(self) -> pd.DataFrame:
        return copy.deepcopy(self._data)

    def _format_pollution_data(self, name: str, pollution_data: pd.DataFrame, year: int) -> pd.DataFrame:
        # Filtering data to only keep relevant Utrecht stations
        stations = ['NL10636', 'NL10639', 'NL10641', 'NL10643']

        for station in stations:
            if station not in pollution_data.columns:
                pollution_data[station] = np.nan

        pollution_data = pollution_data.filter(
            [' Begindatumtijd'] + stations)

        pollution_data['Day'] = pollution_data[' Begindatumtijd'].str[:10]

        pollution_data_means = pollution_data.groupby('Day').agg(
            {'NL10636': 'mean', 'NL10639': 'mean', 'NL10641': 'mean', 'NL10643': 'mean'})

        pollution_data_per_day = pd.DataFrame(
            columns=['Day', 'NL10636', 'NL10639', 'NL10641', 'NL10643', f"Average_{name}"])

        for _, day in enumerate(pollution_data_means.index):
            current_row = {
                'Day': day,
                'NL10636': pollution_data_means.loc[day, 'NL10636'],
                'NL10639': pollution_data_means.loc[day, 'NL10639'],
                'NL10641': pollution_data_means.loc[day, 'NL10641'],
                'NL10643': pollution_data_means.loc[day, 'NL10643'],
                f"Average_{name}": pollution_data_means.loc[day].mean(skipna=True),
            }
            pollution_data_per_day.loc[len(pollution_data_per_day)] = current_row

        pollution_data_per_day.to_csv(
            os.path.join(os.getcwd(), r"data/processed", f"{name},{year}.csv"), index=False)
        return pollution_data_per_day

    def _extract_timeframe(self, starting_year: int, end_year: int) -> None:
        # Note: there are no missing values between 19931231 and 20240912
        starting_date = str(starting_year) + "0101"
        end_date = str(end_year) + "1231"

        # Selecting climate data in given timeframe from CSV file
        raw_data_file = open(os.path.join(os.getcwd(), "data/raw", "climate_data.txt"), "r")
        processed_data_file = open(
            os.path.join(os.getcwd(), r"data/processed", f"climate,{starting_year}-{end_year}.csv"), "w")

        reading_data = False

        for line in raw_data_file:
            if re.match(r'.*{}.*'.format(starting_date), line):
                reading_data = True
            if re.match(r'#.*', line) or reading_data:
                processed_data_file.writelines(line.replace(" ", ""))
            if re.match(r'.*{}.*'.format(end_date), line):
                break

        raw_data_file.close()
        processed_data_file.close()

        self._climate_data = pd.read_csv(
            os.path.join(os.getcwd(), r"data/processed", f"climate,{starting_year}-{end_year}.csv"))

    def _format_legacy_pollution_dates(self, date: str):
        return (
                date[:4] + "-" +  # Year
                date[4:6] + "-" +  # Month
                date[6:8] + "T" +  # Day
                date[9:] + ":00+01:00"  # Time and timezone
        )

    def format_data(self, starting_year: int, ending_year: int) -> None:
        """_summary_

        Args:
            starting_year (int): _description_
            ending_year (int): _description_
        """
        assert isinstance(starting_year, int)
        assert isinstance(ending_year, int)
        assert starting_year <= ending_year

        no2_pollution_data = None
        o3_pollution_data = None

        for year in range(starting_year, ending_year + 1):
            no2_pollution_data_year: DataFrame = pd.read_csv(os.path.join(os.getcwd(), "data/raw", f"{year}_NO2.csv"),
                                                             delimiter=';', encoding="ISO-8859-1", skiprows=9)
            o3_pollution_data_year: DataFrame = pd.read_csv(os.path.join(os.getcwd(), "data/raw", f"{year}_O3.csv"),
                                                            delimiter=';', encoding="ISO-8859-1", skiprows=9)

            if year < 2023:  # Different format is used for csv files before 2023
                no2_pollution_data_year[' Begindatumtijd'] = no2_pollution_data_year[' Begindatumtijd'].apply(
                    self._format_legacy_pollution_dates)
                no2_pollution_data_year['Einddatumtijd'] = no2_pollution_data_year['Einddatumtijd'].apply(
                    self._format_legacy_pollution_dates)
                o3_pollution_data_year[' Begindatumtijd'] = o3_pollution_data_year[' Begindatumtijd'].apply(
                    self._format_legacy_pollution_dates)
                o3_pollution_data_year['Einddatumtijd'] = o3_pollution_data_year['Einddatumtijd'].apply(
                    self._format_legacy_pollution_dates)

            no2_pollution_data_year = self._format_pollution_data("no2", no2_pollution_data_year, year)
            no2_pollution_data = pd.concat([no2_pollution_data, no2_pollution_data_year], ignore_index=True)
            o3_pollution_data_year = self._format_pollution_data("o3", o3_pollution_data_year, year)
            o3_pollution_data = pd.concat([o3_pollution_data, o3_pollution_data_year], ignore_index=True)

        self._no2_pollution_data = no2_pollution_data
        no2_pollution_data.to_csv(os.path.join(os.getcwd(), r"data/processed", "no2.csv"), index=False)
        self._o3_pollution_data = o3_pollution_data
        o3_pollution_data.to_csv(os.path.join(os.getcwd(), r"data/processed", "o3.csv"), index=False)

        # Formatting climate data
        self._extract_timeframe(starting_year, ending_year)
        self._climate_data = self._climate_data.drop(columns="#STN")

        # Putting data files together
        data = pd.concat([self._climate_data, no2_pollution_data["Average_no2"], o3_pollution_data["Average_o3"]],
                         axis=1,
                         join="inner")
        data.to_csv(os.path.join(os.getcwd(), r"data/processed", f"data,{starting_year}-{ending_year}.csv"),
                    index=False)

        self._data = data
