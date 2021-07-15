import torch
import torch.nn as nn
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(32,16)
    
    def forward(self, x):
        return self.fc(x)


if __name__ == '__main__':
    model = MyModel()
    x = 
    