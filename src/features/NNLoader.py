import pandas as pd
import torch
from torch.utils.data import Dataset


# A custom dataset s.t. you can train the model on
class NNLoader(Dataset):
    def __init__(self, feature_data: pd.DataFrame, label_data: pd.DataFrame, window_size: int = 2) -> None:
        super().__init__()
        # Combine features and (past) labels as the input
        # This is not implemented in feature_engineering as of writing this
        self._features = pd.concat([feature_data.reset_index(drop=True), label_data.reset_index(drop=True)],
                                   join="inner", axis=1).reset_index(drop=True)
        self._label_data = label_data.reset_index(drop=True)
        self._timeframe = window_size

    def __len__(self) -> int:
        return len(self._features.index)

    def __getitem__(self, index) -> tuple[torch.tensor, torch.tensor]:
        expected_dates = [self._features.iloc[index]["YYYYMMDD"] + i for i in range(self._timeframe + 4)]
        try:
            obtained_dates = [self._features.iloc[i+index]["YYYYMMDD"] for i in range(self._timeframe + 4)]
        except Exception:
            obtained_dates = 0
        features_tensor = torch.tensor([1e9])
        labels_tensor = torch.tensor([1e9])

        if obtained_dates == expected_dates:
            # Allow for size of timeframe
            features = self._features.iloc[index:index + self._timeframe]
            # Move y outcome along as well
            labels = self._label_data.iloc[index + self._timeframe:index + self._timeframe + 4]

            features = features.drop(columns="YYYYMMDD")
            features_tensor = torch.tensor(features.values)
            labels_tensor = torch.tensor(labels.values)

        return features_tensor, labels_tensor
