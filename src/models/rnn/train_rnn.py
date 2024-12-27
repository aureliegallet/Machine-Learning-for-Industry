from src.models.rnn.RNN import RNN
import torch
from torch.utils.data import DataLoader
import torch.utils.tensorboard as tensorboard
from datetime import datetime
from torchmetrics import MeanAbsolutePercentageError
from tqdm import tqdm

timestamp = datetime.now().strftime('%H_%M')
writer = tensorboard.SummaryWriter('runs/rnn_{}'.format(timestamp))


class CreateTrainRNN:
    def __init__(self, training_data: DataLoader, validation_data: DataLoader, learning_rate=0.01) -> None:
        self.model = RNN(43, 64, 2)
        self._optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate)
        self._loss_fn = MeanAbsolutePercentageError()
        # Scheduler reduces the learning rate by a factor of gamma each step_size steps
        self._scheduler = torch.optim.lr_scheduler.StepLR(self._optimizer, step_size=13, gamma=0.7)

        self._training_loader = training_data
        self._validation_loader = validation_data

    def _train_one_epoch(self) -> float:
        """Trains the model on one epoch. Computes average loss per batch, and reports this.

        Args:
            epoch_index (int): To keep track of the amount of data used (?)

        Returns:
            float: The last average loss
        """

        running_loss = 0
        last_loss = 0
        report_per_samples = 100  # Specifies per how many samples you want to report
        for i, data in enumerate(self._training_loader):
            inputs, labels = data
            if torch.isnan(inputs).any() or 1e9 in labels:
                continue
            self._optimizer.zero_grad()
            outputs = self.model(inputs)

            if outputs.shape != labels.shape:
                continue

            loss = self._loss_fn(outputs, labels)
            loss.backward()

            self._optimizer.step()

            # Gather data and report
            running_loss += loss.item()
            if i % report_per_samples == report_per_samples - 1:
                last_loss = running_loss / report_per_samples
                print('  batch {} loss: {}'.format(i + 1, last_loss))
                running_loss = 0

        return last_loss

    def train_rnn(self, epochs=100) -> None:
        best_vloss = 99999999
        for epoch in tqdm(range(epochs), desc="Training Progress", unit="epoch"):

            self.model.train(True)
            average_loss = self._train_one_epoch(epoch)

            running_vloss = 0.0
            with torch.no_grad():
                self.model.eval()
                for i, data in enumerate(self._validation_loader):
                    vinputs, vlabels = data
                    # Some data inputs contain nan, you do not want to input this into the model
                    if torch.isnan(vinputs).any() or 1e9 in vlabels:
                        continue

                    voutputs = self.model(vinputs)

                    if voutputs.shape != vlabels.shape:
                        continue

                    vloss = self._loss_fn(voutputs, vlabels)
                    running_vloss += vloss

            average_vloss = running_vloss / (i+1)
            print('LOSS train {} valid {}'.format(average_loss, average_vloss))
            writer.add_scalars('Loss',
                               {'Training': average_loss, 'Validation': average_vloss},
                               epoch + 1)

            if average_vloss < best_vloss:
                best_vloss = average_vloss
                model_path = 'runs/rnn_{}/model_{}.pt'.format(timestamp, epoch)
                torch.save(self.model, model_path)

        torch.save(self.model, "results/rnn/rnn_model")

        self._scheduler.step()
        writer.flush()
