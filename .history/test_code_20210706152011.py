import torch
import torch.nn as nn
import numpy as np

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10,16)
    
    def forward(self, x):
        return self.fc(x)


if __name__ == '__main__':
    model = MyModel()
    x = torch.randn(3,10)
    y = model(x)

    