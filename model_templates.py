from torchvision import models
import torch.nn as nn
import torch
import numpy as np


class LSTMModel(nn.Module):
    def __init__(
        self, input_size, hidden_size, num_layers, hidden_linear_size
    ):
        """
        input_size: size of input features
        hidden_size: nodes in hidden layer / size of output features
        num_layers: number of stacked LSTM layers
        hidden_linear_size: in regression
        """
        
        super(LSTMModel, self).__init__()

        # Main LSTM
        self.lstm = nn.LSTM(input_size=input_size, 
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=False)

        # Regressor
        self.regression = nn.Sequential(
            nn.Linear(hidden_size, hidden_linear_size),
            nn.ReLU(),
            nn.Linear(hidden_linear_size, 1)
        )

    def forward(self, data):
        i_lstm, _ = self.lstm(data[0])

        # Feed concatenated outputs into the
        # regession networks.
        prediction = torch.squeeze(self.regression(i_lstm[-1]))
        return prediction


class LinearRegressionModel(nn.Module):
    def __init__(self, input_size, hidden_linear_size): 
        super(LinearRegressionModel, self).__init__() 
        self.regression = nn.Sequential(
            nn.Linear(input_size, hidden_linear_size),
            nn.ReLU(),
            nn.Linear(hidden_linear_size, 1)
        )
  
    def forward(self, x): 
        y_pred = torch.squeeze(self.regression(x))
        return y_pred
