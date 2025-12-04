from load_data import BODMASLoader
import os


file_path = os.path.join(os.path.dirname(__file__), "..", "metadata", "bodmas.npz")
zero_count = 40000
one_count = 40000
test_size = 0.2
loader = BODMASLoader(file_path)
loader.load()
X_sub, y_sub = loader.sample_subset(zero_count=zero_count, one_count=one_count)
X_train, X_test, y_train, y_test = loader.split(X_sub, y_sub, test_size=test_size)

import torch
from torch.utils.data import TensorDataset, DataLoader

X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.LongTensor(y_train)
X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.LongTensor(y_test)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

batch_size = 128
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

import torch.nn as nn


class DNN(nn.Module):
    def __init__(self, input_size=2381):
        super(DNN, self).__init__()
        self.hidden_sizes = [1024, 512, 256]
        self.num_classes = 2
        layers = []
        prev_size = input_size

        for hidden_size in self.hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.5))
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, self.num_classes))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)