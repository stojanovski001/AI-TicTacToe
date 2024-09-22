import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, device, num_layers=10, num_neurons=1024):
        super(DQN, self).__init__()
        self.device = device

        # Create a list to hold layers
        layers = [nn.Linear(9, num_neurons)]
        layers.append(nn.ReLU())
        
        # Create hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(num_neurons, num_neurons))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=0.2))  # Dropout after each hidden layer

        # Output layer
        layers.append(nn.Linear(num_neurons, 9))
        
        # Register the layers
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        x = x.to(self.device)  # Move input tensor to the device
        return self.network(x)
