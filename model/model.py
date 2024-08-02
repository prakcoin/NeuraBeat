import torch
import torch.nn as nn
from .residualblock import ResidualBlock
    
class EmbeddingModel(nn.Module):
    def __init__(self):
      super(EmbeddingModel, self).__init__()
      self.input_layer = nn.Sequential(
          nn.Conv2d(in_channels=1, out_channels=8, kernel_size=5, bias=False),
          nn.SELU(),
      )

      self.conv_layers = nn.Sequential(
          ResidualBlock(in_channels=8, out_channels=8, kernel_size=3, stride=2, num_layers=2),
          ResidualBlock(in_channels=8, out_channels=16, kernel_size=3, stride=2, num_layers=2),
          ResidualBlock(in_channels=16, out_channels=32, kernel_size=3, stride=2, num_layers=2),
          ResidualBlock(in_channels=32, out_channels=64, kernel_size=3, stride=2, num_layers=2),
      )

      self.dense_layers = nn.Sequential(
          nn.Linear(in_features=64, out_features=256),
          nn.ReLU(),
          nn.Linear(in_features=256, out_features=128),
          nn.ReLU(),
          nn.Linear(in_features=128, out_features=128)
      )

    def forward(self, x):
      x = self.input_layer(x)
      x = self.conv_layers(x)
      x = torch.mean(x.view(x.size(0), x.size(1), -1), dim=2)
      x = self.dense_layers(x)
      return x

    def get_embedding(self, x):
      return self.forward(x)