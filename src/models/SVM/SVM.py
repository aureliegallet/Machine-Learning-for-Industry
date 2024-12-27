import numpy as np
import pandas as pd
from sklearn import svm
from src.data.data_storer import DataStorer
from sklearn.multioutput import MultiOutputRegressor


# Ensemble class where each SVM predicts a different day based on different pca_components
# (from random forest inspection)
class SVMEnsemble:
    def __init__(self, ) -> None:
        # Create all three SVMs
        self._SVMs = []
        for _ in range(3):
            svr = svm.SVR(gamma='scale')
            multi_output_svr = MultiOutputRegressor(svr)
            self._SVMs.append(multi_output_svr)

    def train(self, data: list[DataStorer]) -> None:
        assert len(data) == 3  # because you predict three days
        for data_part, model in zip(data, self._SVMs):
            assert isinstance(data_part, DataStorer)
            current_train_x = data_part.train_x.to_numpy()
            current_train_y = data_part.train_y.to_numpy()
            model.fit(current_train_x, current_train_y)

    def predict(self, data: list[pd.DataFrame]) -> list[np.ndarray]:
        assert isinstance(data, list)
        y_output = []
        for model, input_data in zip(self._SVMs, data):
            assert isinstance(input_data, pd.DataFrame)
            y_pred = model.predict(input_data.to_numpy())
            y_output.append(y_pred)
        return y_output
