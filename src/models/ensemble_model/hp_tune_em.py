from typing import Any

from datetime import datetime

from torch.utils.data import DataLoader
from torchmetrics import MeanAbsolutePercentageError
import umap
from src.data.data_storer import DataStorer
from src.models.ensemble_model.basic_lstm import BasicLSTM
from src.models.ensemble_model.ensemble_model import EnsembleModel
from src.features.HP_data_loader import HPDataLoader
from torchmetrics import MeanSquaredError
from torchmetrics import R2Score
from src.features.NNLoader import NNLoader
from src.features.feature_engineering import FeatureEngineering
import random
import pandas as pd
import os
import tempfile
from pathlib import Path
import torch
import torch.nn as nn
from ray import tune
from ray import train
from ray.train import Checkpoint, get_checkpoint
from ray.tune.schedulers import ASHAScheduler
import ray.cloudpickle as pickle
import pickle as pk
from sklearn.decomposition import PCA

device = "cpu"
if torch.cuda.is_available():
    device = "cuda:0"

# Put in main for ease of access
timestamp = datetime.now().strftime('%H_%M')
loss_fn = MeanAbsolutePercentageError().to(device)
mse = MeanSquaredError().to(device)
r2 = R2Score().to(device)


def hyperparameter_tuning_ensemble(data_loader: DataStorer, data_storer: DataStorer, max_epochs: int = 100,
                                   attempts: int = 10, window_size: int = 2) -> None:
    def sample_num_models(config):
        minimum_models = (int(config["num_features"] / config["input_size_models"])) + 1
        return random.randint(minimum_models, minimum_models + 10)

    def sample_num_features(config):
        match config["feature_engineering"]:
            case "umap":
                return random.randint(17, 32)
            case "pca":
                return 43
            case _:
                raise ValueError()  # This should never have to be raised

    def sample_input_size_models(config):
        return random.randint(1, config["num_features"])

    config = {
        "feature_engineering": tune.choice(["umap", "pca"]),
        "num_features": tune.sample_from(lambda spec: sample_num_features(spec.config)),
        "input_size_models": tune.sample_from(lambda spec: sample_input_size_models(spec.config)),
        "num_models": tune.sample_from(lambda spec: sample_num_models(spec.config)),
        "hidden_size": tune.choice([(i * 5 + 5) for i in range(6)]),
        "num_layers": tune.choice([1, 2, 3]),
        "lr": tune.loguniform(1e-4, 1e-1),
    }
    # Terminates models early if bad performance
    scheduler = ASHAScheduler(max_t=max_epochs, grace_period=10, reduction_factor=2)
    gpus_per_trial = 1
    tuner = tune.Tuner(tune.with_resources(
        tune.with_parameters(lambda config: train_ensemble_model(config, data_loader=data_loader)),
        resources={"cpu": 2, "gpu": 1}
    ),
        tune_config=tune.TuneConfig(
            metric="val_loss",
            mode="min",
            scheduler=scheduler,
            num_samples=attempts,
            time_budget_s=27000
        ),
        param_space=config
    )
    result = tuner.fit()

    best_trial = result.get_best_result("val_loss", "min", "last")
    print(f"Best trial config: {best_trial.config}")
    print(f"Best trial final validation loss: {best_trial.metrics['val_loss']}")
    print(f"Best trial final validation MSE: {best_trial.metrics['MSE']}")
    print(f"Best trial final validation R2: {best_trial.metrics['R2']}")

    best_checkpoint = best_trial.get_best_checkpoint(metric="val_loss", mode="max")
    best_trained_model = None
    with best_checkpoint.as_directory() as checkpoint_dir:
        data_path = Path(checkpoint_dir) / "data.pkl"
        print("Getting model data at {}".format(checkpoint_dir))
        with open(data_path, "rb") as fp:
            best_checkpoint_data = pickle.load(fp)
            best_trained_model = best_checkpoint_data["model"]

        transf_path = Path(checkpoint_dir) / "transformation_object.pkl"
        with open(transf_path, "rb") as f:
            transformation_object = pk.load(f)

        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda:0"
            if gpus_per_trial > 1:
                best_trained_model = nn.DataParallel(best_trained_model)
        best_trained_model.to(device)

        MAPE, MSE, R2 = test_accuracy(best_trained_model, data_storer, config, transformation_object)
        print("Best MAPE, MSE and R2 score: {}, {}, {}".format(MAPE, MSE, R2))

    os.makedirs("best_model", exist_ok=True)
    torch.save(best_trained_model, "best_model/best_model_{}".format(timestamp))
    print("Saved final model!")


def train_ensemble_model(config: dict, data_loader: HPDataLoader, epochs=1) -> None:
    learning_rate = config["lr"]

    checkpoint = get_checkpoint()
    transformation_object = None

    if checkpoint:
        with checkpoint.as_directory() as checkpoint_dir:
            data_path = Path(checkpoint_dir) / "data.pkl"
            with open(data_path, "rb") as fp:
                checkpoint_state = pickle.load(fp)
            start_epoch = checkpoint_state["epoch"]
            model = checkpoint_state["model"]
            optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
            optimizer.load_state_dict(checkpoint_state["optimizer_state_dict"])
            transf_path = Path(checkpoint_dir) / "transformation_object.pkl"
            with open(transf_path, "rb") as f:
                transformation_object = pk.load(f)

    else:
        model = create_model(config["num_features"], config["num_models"], config["input_size_models"],
                             config["hidden_size"], config["num_layers"])
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        start_epoch = 0

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
    model.to(device)

    # Get correct data (with feature engineering)
    feature_engineering = FeatureEngineering(3)

    # Remove dates
    data_storer = data_loader.data_stored
    date_train = data_storer.train_x['YYYYMMDD']
    data_storer.train_x = data_storer.train_x.drop(columns="YYYYMMDD")
    date_val = data_storer.val_x['YYYYMMDD']
    data_storer.val_x = data_storer.val_x.drop(columns="YYYYMMDD")

    # Apply feature engineering and store the created object in ensemble model
    if config["feature_engineering"] == "pca":
        if transformation_object is not None:
            pca_object = PCA()
            pca_object = transformation_object

        else:
            pca_object = feature_engineering.create_pca(data_storer.train_x)
        train_x = pd.DataFrame(data=pca_object.transform(data_storer.train_x),
                               columns=[f"pca_component_{i + 1}" for i in range(config["num_features"] - 2)])
        val_x = pd.DataFrame(data=pca_object.transform(data_storer.val_x),
                             columns=[f"pca_component_{i + 1}" for i in range(config["num_features"] - 2)])
        transformation_object = pca_object
    else:  # If umap
        if transformation_object is not None:
            umap_object = umap.UMAP()
            umap_object = transformation_object
        else:
            umap_object = feature_engineering.create_umap(data_storer.train_x, config["num_features"] - 2)
        train_x = pd.DataFrame(data=umap_object.transform(data_storer.train_x),
                               columns=[f"umap_component_{i + 1}" for i in range(config["num_features"] - 2)])
        val_x = pd.DataFrame(data=umap_object.transform(data_storer.val_x),
                             columns=[f"umap_component_{i + 1}" for i in range(config["num_features"] - 2)])
        transformation_object = umap_object

    # Add dates back in
    train_x = pd.concat([train_x.reset_index(drop=True), date_train.reset_index(drop=True)], axis=1, join="inner")
    val_x = pd.concat([val_x.reset_index(drop=True), date_val.reset_index(drop=True)], axis=1, join="inner")

    training_data, validation_data = data_loader.load_data_loader(train_x, data_storer.train_y, val_x,
                                                                  data_storer.val_y)

    for epoch in range(start_epoch, epochs):
        # Training part
        model.train(True)
        average_loss = train_one_epoch(epoch, training_data, optimizer, model)

        # Evaluation part
        model.eval()
        running_vloss = 0.0
        running_mse = 0.0
        running_r2 = 0.0
        with torch.no_grad():
            model.eval()
            for i, data in enumerate(validation_data):
                vinputs, vlabels = data
                vinputs, vlabels = vinputs.to(device), vlabels.to(device)

                # Some data inputs contain nan, you do not want to input this into the model
                if torch.isnan(vinputs).any() or 1e9 in vlabels:
                    continue

                # Model splits data itself
                voutputs = model(vinputs)
                voutputs, _ = torch.median(voutputs.float(), dim=0)

                # This splits output into a list of items (I think?)
                split_voutputs = torch.split(voutputs, 1, dim=1)

                # Labels are also split
                split_vlabels = torch.split(vlabels, 1, dim=1)

                if len(split_voutputs) != len(split_vlabels):
                    continue

                running_vloss += loss_fn(voutputs, vlabels)
                running_mse += mse(voutputs.flatten(), vlabels.flatten())
                running_r2 += r2(voutputs.flatten(), vlabels.flatten())

        checkpoint_data = {
            "epoch": epoch,
            "model": model,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }

        with tempfile.TemporaryDirectory() as checkpoint_dir:
            data_path = Path(checkpoint_dir) / "data.pkl"
            with open(data_path, "wb") as fp:
                pickle.dump(checkpoint_data, fp)

            transf_path = Path(checkpoint_dir) / "transformation_object.pkl"
            with open(transf_path, "wb") as f:
                pk.dump(transformation_object, f)

            checkpoint = Checkpoint.from_directory(checkpoint_dir)
            train.report(
                {"train_loss": float(average_loss), "val_loss": float(running_vloss / (i + 1)),
                 "MSE": float(running_mse / (i + 1)), "R2": float(running_r2 / (i + 1))},
                checkpoint=checkpoint,
            )

    print(f"Stored results in {checkpoint_dir}")
    print("Finished task")


def create_model(num_features: int, num_models: int, input_size_models: int, hidden_size: int,
                 num_layers: int) -> EnsembleModel:
    """Creates ensemble model based on different parameters.
    Each LSTM model consists of a random amount of features, all features get used in the models.

    Args:
        num_features (int): The amount of features into the ensemble model
        num_models (int): The amount of models
        input_size_models (int): The input size of each ensemble model
        hidden_size (int): _description_
        num_layers (int): _description_

    Returns:
        EnsembleModel: The ensemble model of LSTMs
    """
    if num_features < input_size_models:
        raise ValueError()

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"

    # Shuffle the feature set into new sets
    feature_division_list = [[] for _ in range(num_models)]
    features_as_list = [i for i in range(num_features)]
    shuffled_numbers = random.sample(features_as_list, len(features_as_list))
    for i, num in enumerate(shuffled_numbers):
        feature_division_list[i % num_models].append(num)

    # Append extra features s.t. models have inputs = num_features (bit inefficient, but that's alright)
    for lst in feature_division_list:
        while len(lst) != input_size_models:
            sample = random.sample(shuffled_numbers, k=1)
            if sample[0] not in lst:
                lst.append(sample[0])

    # Create model
    models = []
    for i in range(num_models):
        models.append(
            BasicLSTM(input_size=input_size_models, hidden_size=hidden_size, num_layers=num_layers, out_features=2,
                      device=device))
    model = EnsembleModel(models, 2, feature_division_list)
    return model


def test_accuracy(model: Any, data_storer: DataStorer, transformation_object: Any) -> tuple[float, float, float]:
    # Create testing data (applying transformation as well)
    date_val = data_storer.test_x['YYYYMMDD']
    data_storer.test_x = data_storer.test_x.drop(columns="YYYYMMDD")
    test_x = pd.DataFrame(data=transformation_object.transform(data_storer.test_x))
    test_x = pd.concat([test_x.reset_index(drop=True), date_val.reset_index(drop=True)], axis=1, join="inner")
    testing_data = NNLoader(test_x, data_storer.test_y, window_size=2)
    testing_data = DataLoader(testing_data, batch_size=1, shuffle=False)

    running_mape = 0
    running_mse = 0
    running_r2 = 0

    with torch.no_grad():
        model.eval()
        for i, data in enumerate(testing_data):
            vinputs, vlabels = data
            vinputs, vlabels = vinputs.to(device), vlabels.to(device)
            if torch.isnan(vinputs).any() or 1e9 in vlabels:
                continue
            voutputs = model(vinputs)
            voutputs, _ = torch.median(voutputs.float(), dim=0)
            split_voutputs = torch.split(voutputs, 1, dim=1)
            split_vlabels = torch.split(vlabels, 1, dim=1)
            if len(split_voutputs) != len(split_vlabels):
                continue
            running_mape += loss_fn(voutputs, vlabels)  # Loss and mape are the same in this case
            running_mse += mse(voutputs.flatten(), vlabels.flatten())
            running_r2 += r2(voutputs.flatten(), vlabels.flatten())

    return ((running_mape / (i + 1)), (running_mse / (i + 1)), (running_r2 / (i + 1)))


def train_one_epoch(epoch_index: int, train_data_loader: DataLoader, optimizer: Any, model: Any) -> float:
    """Trains the model on one epoch. Computes average loss per batch, and reports this.

    Args:
        epoch_index (int): To keep track of the amount of data used (?)

    Returns:
        float: The last average loss
    """

    # Very similar to train_em, see that function to understand here
    running_loss = 0
    last_loss = 0
    report_per_samples = 100  # Specifies per how many samples you want to report

    for i, data in enumerate(train_data_loader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        if torch.isnan(inputs).any() or 1e9 in labels:
            continue

        optimizer.zero_grad()
        outputs = model(inputs)
        outputs, _ = torch.median(outputs.float(), dim=0)
        split_outputs = torch.split(outputs, 1, dim=1)
        split_labels = torch.split(labels, 1, dim=1)
        if len(split_outputs) != len(split_labels):
            continue

        loss = loss_fn(outputs, labels)
        loss.backward()

        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if i % report_per_samples == report_per_samples - 1:
            last_loss = running_loss / report_per_samples
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            running_loss = 0
    return last_loss
