import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from src.data.data_storer import DataStorer


def feature_selection_rf(x_train: DataStorer, y_train: DataStorer, feature_selection_threshold=0.02) -> list:
    """ Apply random forests and evaluate them based on mean decrease in impurity (measure for random forests),
        show this in a plot
    """
    feature_names = [f"component_{i+1}" for i in range(41)]
    rf = RandomForestRegressor(random_state=0)
    rf.fit(x_train, y_train)
    importances = rf.feature_importances_
    forest_importances = pd.Series(importances, index=feature_names)
    integer_list = [int(index.split('_')[1]) for index in
                    forest_importances[importances > feature_selection_threshold].index]
    print(f"PCA components to use as input: {integer_list}")
    return integer_list
