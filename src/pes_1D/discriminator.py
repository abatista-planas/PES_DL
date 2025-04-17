import torch.nn as nn
import torch.nn.functional as F


class NaiveDiscriminator(nn.Module):
    def __init__(self, in_features=4, h1=8, h2=9, h3=10, out_features=3):
        super(NaiveDiscriminator, self).__init__()

        self.fc1 = nn.Linear(in_features, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, h3)
        self.dropout = nn.Dropout(0.3)
        self.out = nn.Linear(h3, out_features)
        pass

    def forward(self, x):
        x = F.relu(self.fc1(x))
        # x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.out(x)
        return x
