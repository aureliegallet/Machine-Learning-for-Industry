import torch
from torch import nn
from torch.autograd import Variable


class BasicLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, out_features: int, output_days: int = 4,
                 device="cpu") -> None:
        super().__init__()
        self.device = device

        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_days = output_days
        self.out_features = out_features

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True).double()
        # output_days days in advance
        self.linear = nn.Linear(in_features=self.hidden_size, out_features=out_features * output_days).double()

    def forward(self, x) -> torch.Tensor:
        # hidden state
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=self.device)).double()
        # internal state
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=self.device)).double()

        # Propagate input through LSTM
        output, (hn, cn) = self.lstm(x, (h_0, c_0))
        hn = hn.view(-1, self.hidden_size)
        final_hidden_state = output[:, -1, :]

        # Transform output into 2 output variables
        out = self.linear(final_hidden_state)
        out = out.view(-1, self.output_days, self.out_features)
        return out
