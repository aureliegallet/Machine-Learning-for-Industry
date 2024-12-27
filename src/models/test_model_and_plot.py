from typing import Any
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from torchmetrics import MeanAbsolutePercentageError
from src.data.data_storer import DataStorer
from torchmetrics import MeanSquaredError
from torchmetrics import R2Score
from src.features.NNLoader import NNLoader
import pandas as pd
import torch
import pickle as pk


def plot_testing_results(o3_predicted: list, no2_predicted: list, o3_actual: list, no2_actual: list, name: str) -> None:
    o3_p1 = []
    o3_p2 = []
    o3_p3 = []
    o3_p4 = []
    no2_p1 = []
    no2_p2 = []
    no2_p3 = []
    no2_p4 = []
    o3_a = []
    no2_a = []
    for i in range(0, len(o3_predicted), 4):
        o3_a.append(o3_actual[i])
        no2_a.append(no2_actual[i])
        o3_p1.append(o3_predicted[i])
        o3_p2.append(o3_predicted[i + 1])
        o3_p3.append(o3_predicted[i + 2])
        o3_p4.append(o3_predicted[i + 3])
        no2_p1.append(no2_predicted[i])
        no2_p2.append(no2_predicted[i + 1])
        no2_p3.append(no2_predicted[i + 2])
        no2_p4.append(no2_predicted[i + 3])
    plot_and_save_image(o3_p1, o3_a, no2_p1, no2_a, 0, len(o3_p1), f"{name} Day 1.png")
    plot_and_save_image(o3_p2, o3_a, no2_p2, no2_a, 0, len(o3_p2), f"{name} Day 2.png")
    plot_and_save_image(o3_p3, o3_a, no2_p3, no2_a, 0, len(o3_p3), f"{name} Day 3.png")
    plot_and_save_image(o3_p4, o3_a, no2_p4, no2_a, 0, len(o3_p4), f"{name} Day 4.png")
    return


def test_data(data_storer: DataStorer, path_model: str = None, path_transf: str = None, model: Any = None,
              transformation_object: Any = None) -> tuple[list, list, list, list]:
    # Small file to test the model
    if path_model is not None:
        with open(path_model, "rb") as f:
            model = torch.load(f)

    if path_transf is not None:
        with open(path_transf, "rb") as f:
            transformation_object = pk.load(f)

    date_val = data_storer.test_x['YYYYMMDD']
    data_storer.test_x = data_storer.test_x.drop(columns="YYYYMMDD")
    test_x = pd.DataFrame(data=transformation_object.transform(data_storer.test_x))
    test_x = pd.concat([test_x.reset_index(drop=True), date_val.reset_index(drop=True)], axis=1, join="inner")

    testing_data = NNLoader(test_x, data_storer.test_y, window_size=2)
    testing_data = DataLoader(testing_data, batch_size=1, shuffle=False)

    device = ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    running_mape = [0, 0, 0, 0, 0]
    running_mse = [0, 0, 0, 0, 0]
    running_r2 = [0, 0, 0, 0, 0]

    loss_fn = MeanAbsolutePercentageError().to(device)
    mse = MeanSquaredError().to(device)
    r2 = R2Score().to(device)

    o3_predicted = []
    o3_actual = []
    no2_predicted = []
    no2_actual = []

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

            for j, out_batch in enumerate(zip(split_voutputs, split_vlabels)):
                output, label = out_batch
                running_mape[j] += loss_fn(output, label)
                running_mse[j] += mse(output.flatten(), label.flatten())
                running_r2[j] += r2(output.flatten(), label.flatten())
                output, label = output.flatten(), label.flatten()
                o3_predicted.append(output[0])
                no2_predicted.append(output[1])
                o3_actual.append(label[0])
                no2_actual.append(label[1])

    print("Results MAPE:")
    print(
        f"Day 1:{running_mape[0] / (i + 1)}, Day 2:{running_mape[1] / (i + 1)}, Day 3:{running_mape[2] / (i + 1)},"
        f"Day 4:{running_mape[3] / (i + 1)}, All days: {sum(running_mape) / (4 * (i + 1))}")
    print("Results MSE:")
    print(
        f"Day 1:{running_mse[0] / (i + 1)}, Day 2:{running_mse[1] / (i + 1)}, Day 3:{running_mse[2] / (i + 1)},"
        f"Day 4:{running_mse[3] / (i + 1)}, All days: {sum(running_mse) / (4 * (i + 1))}")
    print("Results R2:")
    print(
        f"Day 1:{running_r2[0] / (i + 1)}, Day 2:{running_r2[1] / (i + 1)}, Day 3:{running_r2[2] / (i + 1)},"
        f"Day 4:{running_r2[3] / (i + 1)}, All days: {sum(running_r2) / (4 * (i + 1))}")

    return o3_predicted, no2_predicted, o3_actual, no2_actual


def plot_train_val_csv(path: str, output_file: str) -> None:
    assert isinstance(path, str)
    file = pd.read_csv(path)
    file = file[["train_loss", "val_loss"]]
    train, val = file[["train_loss"]], file[["val_loss"]]
    assert len(train) == len(val)
    days = np.arange(0, len(train))

    plt.figure(figsize=(10, 5))
    plt.plot(days, train, label="Train Loss", color="b")
    plt.plot(days, val, label="Val Loss", color="r")
    plt.title("Train Loss vs. Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()


def plot_and_save_image(predicted_o3: list, actual_o3: list, predicted_no2: list, actual_no2: list,
                        days_into_past: int = None, days_into_future: int = None,
                        output_file: str = 'plot.png') -> None:
    assert isinstance(predicted_no2, list)
    assert isinstance(predicted_o3, list)
    assert isinstance(actual_no2, list)
    assert isinstance(actual_o3, list)
    assert len(predicted_no2) == days_into_past + days_into_future
    assert days_into_past is not None or days_into_future is not None

    # Create time axis (previous x days + next 4 days)
    days = np.arange(-days_into_past, days_into_future)  # Adjust range based on your data

    # Plot o3 values
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)  # First subplot for o3
    plt.plot(days, predicted_o3, label='Predicted o3', color='b', linestyle='--')
    plt.plot(days, actual_o3, label='Actual o3', color='g')
    plt.title('o3 Predictions vs Actual')
    plt.xlabel('Days')
    plt.ylabel('o3 Levels')
    plt.axvline(0, color='k', linestyle=':', label='Today')  # Vertical line to separate past/future
    plt.legend()

    # Plot no2 values
    plt.subplot(2, 1, 2)  # Second subplot for no2
    plt.plot(days, predicted_no2, label='Predicted no2', color='r', linestyle='--')
    plt.plot(days, actual_no2, label='Actual no2', color='m')
    plt.title('no2 Predictions vs Actual')
    plt.xlabel('Days')
    plt.ylabel('no2 Levels')
    plt.axvline(0, color='k', linestyle=':', label='Today')  # Vertical line to separate past/future
    plt.legend()

    # Adjust layout
    plt.tight_layout()

    # Save the plot as an image
    plt.savefig(output_file)

    # Close the figure to free memory
    plt.close()
