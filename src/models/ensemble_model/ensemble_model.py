import torch
from torch import nn


# Standard pytorch structure, this for ensemble model
class EnsembleModel(nn.Module):
    def __init__(self, models: list, out_features: int, features_per_model: list[list[int]]) -> None:
        super().__init__()
        self.models = nn.ModuleList(models)
        for i, model in enumerate(self.models):
            features_per_model[i].sort()
            model.fc = nn.Identity()  # Skip the last neural layer of the LSTM
        self.features_per_model = features_per_model
        self.transformation_object = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        output = []
        # Run each dataset with its respective LSTM model
        for model, features in zip(self.models, self.features_per_model):
            x_i = x[:, :, torch.tensor(features)]
            output.append(model(x_i))
        # Stack all outputs and compute mean on both target variables
        output = torch.stack(output)
        # output = torch.mean(output.float(), dim=0)
        return output
