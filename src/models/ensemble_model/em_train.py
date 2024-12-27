import torch.utils.tensorboard as tensorboard
from typing import Any

from torch.utils.data import DataLoader
from torchmetrics import MeanAbsolutePercentageError
from src.features.HP_data_loader import HPDataLoader
from torchmetrics import MeanSquaredError
from torchmetrics import R2Score
import pandas as pd
import torch
import pickle as pk
from sklearn.decomposition import PCA

writer = tensorboard.SummaryWriter('runs/fashion_trainer')
device = "cpu"
mape = MeanAbsolutePercentageError().to(device)
loss_fn = MeanSquaredError().to(device)
r2 = R2Score().to(device)


def train_ensemble_model(data_loader: HPDataLoader, epochs=100) -> None:
    with open("best_model/best_model", "rb") as f:
        model = torch.load(f)
    with open("best_model/transformation_object.pkl", "rb") as f:
        transformation_object = pk.load(f)

    learning_rate = 0.09844186889924886
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    start_epoch = 0

    model.to(device)

    # Remove dates
    data_storer = data_loader.data_stored
    date_train = data_storer.train_x['YYYYMMDD']
    data_storer.train_x = data_storer.train_x.drop(columns="YYYYMMDD")
    date_val = data_storer.val_x['YYYYMMDD']
    data_storer.val_x = data_storer.val_x.drop(columns="YYYYMMDD")

    # Apply feature engineering and store the created object in ensemble model
    pca_object = PCA()
    pca_object = transformation_object
    train_x = pd.DataFrame(data=pca_object.transform(data_storer.train_x),
                           columns=[f"pca_component_{i+1}" for i in range(43 - 2)])
    val_x = pd.DataFrame(data=pca_object.transform(data_storer.val_x),
                         columns=[f"pca_component_{i+1}" for i in range(43 - 2)])
    transformation_object = pca_object

    # Add dates back in
    train_x = pd.concat([train_x.reset_index(drop=True), date_train.reset_index(drop=True)], axis=1, join="inner")
    val_x = pd.concat([val_x.reset_index(drop=True), date_val.reset_index(drop=True)], axis=1, join="inner")

    training_data, validation_data = data_loader.load_data_loader(train_x, data_storer.train_y,
                                                                  val_x, data_storer.val_y)

    # best_vloss = 1e9
    for epoch in range(start_epoch, epochs):
        # Training part
        model.train(True)
        average_loss = train_one_epoch(epoch, training_data, optimizer, model)

        # Evaluation part
        model.eval()
        running_vloss = 0.0
        running_mape = 0.0
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

                running_mape += mape(voutputs, vlabels)
                running_vloss += loss_fn(voutputs.flatten(), vlabels.flatten())
                running_r2 += r2(voutputs.flatten(), vlabels.flatten())
        writer.add_scalars("Metrics and losses", 
                           {"Train_loss": average_loss, "Val_loss": running_vloss/(i+1),
                            "MAPE": running_mape/(i+1), "R2": running_r2/(i+1)},
                            epoch + 1)

    with open("test/model", "wb") as f:
        torch.save(model, f)
    with open("test/transf.pkl", "wb") as f:
        pk.dump(transformation_object, f)
    writer.flush
    print("Finished task")


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
