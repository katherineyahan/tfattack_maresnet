import torch.nn as nn


class TFModel(nn.Module):

    def __init__(self, dropout=0.5):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv1d(in_channels=4, out_channels=256, kernel_size=24, padding='same'), # filters (output space) 256, kernel size: 24
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=256, out_channels=64, kernel_size=12, padding='same'),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool1d(1),
            nn.Flatten()
        )

        dense_block = [
            nn.Linear(in_features=64, out_features=500),
            nn.ReLU(inplace=True)
        ]

        if dropout > 0:
            dense_block.append(nn.Dropout(p=dropout))

        dense_block.append(nn.Linear(in_features=500, out_features=2))

        self.linear_block = nn.Sequential(*dense_block)

    def forward(self, x):
        x = self.conv_block(x)
        x = self.linear_block(x)
        return x


def get_TFAttack1():
    return TFModel()

