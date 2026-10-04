from torch import nn

class PoseANN(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(42,100), nn.ReLU(), nn.Linear(100,100),
                                     nn.ReLU(), nn.Linear(100,4))

    def forward(self, features):
        return self.network(features)
