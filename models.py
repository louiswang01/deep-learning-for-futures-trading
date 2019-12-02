from torchvision import models
import torch.nn as nn
import torch
import numpy as np


class PrototypeModel(nn.Module):
    def __init__(self, input_size, num_layers):
        super(PrototypeModel, self).__init__()

        self.input_size = input_size
        self.num_layers = num_layers
        final_concat_size = 0

        # Main LSTM
        self.lstm = nn.LSTM(input_size=self.input_size,
                            hidden_size=32,
                            num_layers=1,
                            batch_first=False)
        final_concat_size += 32

        # Regressor
        self.regression = nn.Sequential(
            nn.Linear(final_concat_size, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, data):
        i_lstm, _ = self.lstm(data[0])

        # Feed concatenated outputs into the
        # regession networks.
        prediction = torch.squeeze(self.regression(i_lstm[-1]))
        return prediction


class LinearRegressionModel(nn.Module):
    def __init__(self, input_size): 
        super(LinearRegressionModel, self).__init__() 
        self.regression = nn.Sequential(
            nn.Linear(input_size, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
  
    def forward(self, x): 
        y_pred = torch.squeeze(self.regression(x))
        return y_pred 