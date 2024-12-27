import copy

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error
from torch.utils.data import DataLoader as TorchDataLoader

from src.data.data_formatter import DataFormatter
from src.data.data_exploration import DataExplorer, pca_component_ranking_using_random_forest_plot
from src.data.data_preprocessor import PreProcessor
from src.data.data_storer import DataStorer

from src.features.feature_engineering import FeatureEngineering
from src.features.HP_data_loader import HPDataLoader
from src.features.NNLoader import NNLoader

from src.models.ensemble_model.hp_tune_em import hyperparameter_tuning_ensemble
from src.models.linear_regression.linear_regression import RegressionModel
from src.models.rnn.train_rnn import CreateTrainRNN
from src.models.SVM.SVM import SVMEnsemble
from src.models.test_model_and_plot import test_data, plot_testing_results


if __name__ == "__main__":
    # Assemble data into one CSV
    data_formatter = DataFormatter()
    data_formatter.format_data(2010, 2023)

    # Store the data
    data_storer = DataStorer()
    data_storer.data = data_formatter.data

    # Data Inspection
    # data_explorer = DataExplorer(data_storer.data)
    # data_explorer.show_missing_values()
    # data_explorer.analyse_individual_variables()
    # data_explorer.create_dendrogram()
    # data_explorer.correlation_heatmap_hierarchical_clustering(threshold=0.5)
    # data_explorer.pca_analysis()
    # data_explorer.create_plots()

    # Data Preprocessing
    preprocessor = PreProcessor()
    preprocessor(data_storer)

    # Feature Engineering (for hp tuning)
    feature_engineering = FeatureEngineering(3, pca_in_pipeline=True)
    hp_neural_networks_loader, _, _ = feature_engineering(copy.deepcopy(data_storer))
    
    # Feature engineering (for other models)
    feature_engineering = FeatureEngineering(3, pca_in_pipeline=False)
    neural_networks_loader, regression_loader, regression_loader_feature_selected = feature_engineering(data_storer)

    # # Linear Regression
    # regression = RegressionModel()
    # regression(regression_loader_feature_selected, 3)

    # # Inspect pca_components using random_forests
    # rf_data = regression_loader # Method uses same data as regression_loader
    # pca_component_ranking_using_random_forest_plot(rf_data[0].train_x, rf_data[0].train_y)
    # pca_component_ranking_using_random_forest_plot(rf_data[1].train_x, rf_data[1].train_y)
    # pca_component_ranking_using_random_forest_plot(rf_data[2].train_x, rf_data[2].train_y)

    # # SVM model
    # svm_ensemble = SVMEnsemble()
    # svm_ensemble.train(regression_loader_feature_selected)
    # validation_data = []
    # for i in range(3):
    #    validation_data.append(regression_loader_feature_selected[i].val_x)
    # predicted_data = svm_ensemble.predict(validation_data)
    # for i in range(3):
    #    val_y = regression_loader_feature_selected[i].val_y.to_numpy()
    #    pred_y = predicted_data[i]

    #    r2 = r2_score(val_y, pred_y)
    #    print("R² score:", r2)
    
    #    mse = mean_squared_error(val_y, pred_y)
    #    print("Mean Squared Error (MSE):", mse)
    
    #    mape = mean_absolute_percentage_error(val_y, pred_y)
    #    print("Mean Absolute Percentage Error (MAPE):", mape)

    # Create data loaders, you first create a custom Dataset (from pytorch website)
    # after which you can use TorchDataLoader for easy data loading into model (using TorchDataLoader > own method)
    # training_data = TorchDataLoader(NNLoader(neural_networks_loader.train_x, neural_networks_loader.train_y))
    # validation_data = TorchDataLoader(NNLoader(neural_networks_loader.val_x, neural_networks_loader.val_y))

    # # RNN model
    # rnn_model = CreateTrainRNN(training_data, validation_data)
    # rnn_model.train_rnn(epochs=300)

    # # Ensemble model
    # neural_networks_loader = HPDataLoader(hp_neural_networks_loader, 2)
    # hyperparameter_tuning_ensemble(neural_networks_loader, data_storer, attempts=1, window_size=2)

    # Test hp_tuned ensemble model
    o3_predicted, no2_predicted, o3_actual, no2_actual = (
        test_data(data_storer, path_model="results/best_model/best_model",
                  path_transf="results/best_model/transformation_object.pkl"))
    plot_testing_results(o3_predicted, no2_predicted, o3_actual, no2_actual, "Ensemble_model")
    