import torch
import torch.nn as nn
from .positionalencoding import PositionalEncoding2d
from .residualblock import ResidualBlock
    
class ClassificationModel(nn.Module):
    def __init__(self):
      super(ClassificationModel, self).__init__()
      self.input_layer = nn.Sequential(
          nn.Conv2d(in_channels=1, out_channels=64, kernel_size=5, bias=False),
          nn.SELU(),
      )

      self.conv_layers = nn.Sequential(
          ResidualBlock(in_channels=64, out_channels=64, kernel_size=3, num_layers=2, pool=True, short=True),
          ResidualBlock(in_channels=64, out_channels=128, kernel_size=3, num_layers=2, pool=True, short=True),
          ResidualBlock(in_channels=128, out_channels=256, kernel_size=3, num_layers=4, pool=True, short=True),
          ResidualBlock(in_channels=256, out_channels=512, kernel_size=3, num_layers=4, pool=True, short=True),
          nn.Dropout2d(0.2),
      )

      self.positional_encoding = PositionalEncoding2d(64, 128, 130)
      self.attention = nn.MultiheadAttention(embed_dim=16, num_heads=2, dropout=0.5, batch_first=True)

      self.dense_layers = nn.Sequential(
          nn.Linear(in_features=512, out_features=200, bias=False),
          nn.SELU(),
          nn.Linear(in_features=200, out_features=100, bias=False),
          nn.SELU(),
          nn.Linear(in_features=100, out_features=50, bias=False),
          nn.SELU(),
          nn.Dropout(0.5),
      )

      self.output = nn.Linear(50, 8)

    def forward(self, x):
      x = self.input_layer(x)
      x = self.conv_layers(x)
      x = torch.mean(x.view(x.size(0), x.size(1), -1), dim=2)
      x = self.dense_layers(x)
      out = self.output(x)
      return out
    
class EmbeddingModel(nn.Module):
    def __init__(self):
      super(EmbeddingModel, self).__init__()
      self.input_layer = nn.Sequential(
          nn.Conv2d(in_channels=1, out_channels=64, kernel_size=5, bias=False),
          nn.SELU(),
      )

      self.conv_layers = nn.Sequential(
          ResidualBlock(in_channels=64, out_channels=64, kernel_size=3, num_layers=2, pool=True, short=True),
          ResidualBlock(in_channels=64, out_channels=128, kernel_size=3, num_layers=2, pool=True, short=True),
          ResidualBlock(in_channels=128, out_channels=256, kernel_size=3, num_layers=2, pool=True, short=True),
          nn.Dropout2d(p=0.5),
      )

      self.dense_layers = nn.Sequential(
          nn.Linear(in_features=256, out_features=512, bias=False),
          nn.SELU(),
          nn.Linear(in_features=512, out_features=256, bias=False),
          nn.SELU(),
          nn.Dropout(p=0.7),
      )

      self.output = nn.Linear(in_features=256, out_features=128)

    def forward(self, x):
      x = self.input_layer(x)
      x = self.conv_layers(x)
      x = torch.mean(x.view(x.size(0), x.size(1), -1), dim=2)
      x = self.dense_layers(x)
      out = self.output(x)
      return out

    def get_embedding(self, x):
      return self.forward(x)