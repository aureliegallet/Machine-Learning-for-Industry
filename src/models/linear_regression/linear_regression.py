import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

from src.data.data_storer import DataStorer
from sklearn.metrics import mean_absolute_percentage_error


class RegressionModel:
    def __call__(self, regression_loader: list[DataStorer], days_to_predict: int) -> None:
        """Applies linear_regression as described in the report.

        Returns:
            float: The linear_regression score
        """
        for delay in range(days_to_predict):
            self._make_model_and_score(regression_loader[delay].train_x, regression_loader[delay].train_y,
                                       regression_loader[delay].val_x, regression_loader[delay].val_y)

    def _make_model_and_score(self, train_x: pd.DataFrame, train_y: pd.DataFrame, val_x: pd.DataFrame,
                              val_y: pd.DataFrame) -> tuple[float, float]:
        # Linear reg
        model = LinearRegression()
        model.fit(train_x, train_y)
        pred_y = model.predict(val_x)

        r2 = r2_score(val_y, pred_y)
        print("R² score:", r2)

        mse = mean_squared_error(val_y, pred_y)
        print("Mean Squared Error (MSE):", mse)

        mape = mean_absolute_percentage_error(val_y, pred_y)
        print("Mean Absolute Percentage Error (MAPE):", mape)

        return [r2, mse]
