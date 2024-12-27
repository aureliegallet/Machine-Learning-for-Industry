import torch
import torch.nn as nn


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size) -> None:
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size * 3)

    def forward(self, x) -> torch.Tensor:
        h0 = torch.zeros(1, x.size(0), self.hidden_size)
        out, hn = self.rnn(x.float(), h0)
        out = self.fc(out[:, -1, :])
        out = out.view(-1, 3, 2)
        return out
