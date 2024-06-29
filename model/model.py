import torch
import torch.nn as nn
import torch.nn.functional as F
from separableconv import SeparableConv2d
from positionalencoding import PositionalEncoding2d
from residualblock import ResidualBlock 
    
class ClassificationModel(nn.Module):
    def __init__(self):
      super(ClassificationModel, self).__init__()
      self.input = nn.Sequential(SeparableConv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
                                 nn.SELU())
      self.positional_encoding = PositionalEncoding2d(64, 128, 130)


      self.conv_layers = nn.Sequential(
          ResidualBlock(in_channels=64, out_channels=64, kernel_size=3, num_layers=4, pool=True, short=True, two_dim=True),
          ResidualBlock(in_channels=64, out_channels=128, kernel_size=3, num_layers=4, pool=True, short=True, two_dim=True),
          ResidualBlock(in_channels=128, out_channels=256, kernel_size=3, num_layers=4, pool=True, short=True, two_dim=True),
          ResidualBlock(in_channels=256, out_channels=512, kernel_size=3, num_layers=4, pool=True, short=True, two_dim=True),
      )

      self.attention = nn.MultiheadAttention(embed_dim=16, num_heads=2, dropout=0.5, batch_first=True)

      self.dense_layers = nn.Sequential(
          nn.Linear(in_features=512, out_features=512, bias=False),
          nn.SELU(),
          nn.Linear(in_features=512, out_features=256, bias=False),
          nn.SELU(),
          nn.Linear(in_features=256, out_features=128, bias=False),
          nn.SELU(),
          nn.Dropout(0.5),
      )

      self.output = nn.Linear(128, 8)

    def forward(self, x):
      x = self.input(x)
      # x = F.layer_norm(self.positional_encoding(x), x.shape)
      x = self.conv_layers(x)

      # batch_size, channels, height, width = x.size()
      # x = x.view(batch_size, channels, height * width)
      # attention_output, _ = self.attention(x, x, x)
      # x = F.layer_norm(x + attention_output, x.shape)

      x = torch.mean(x.view(x.size(0), x.size(1), -1), dim=2)
      x = self.dense_layers(x)
      out = self.output(x)
      return out

    def get_embedding(self, x):
      return self.forward(x)