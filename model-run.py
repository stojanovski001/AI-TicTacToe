import torch
import torch.nn as nn
import torch.nn.functional as F
from torchviz import make_dot

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

        # To store activations from hooks
        self.activations = {}

    def forward(self, x):
        x = x.to(self.device)  # Move input tensor to the device
        return self.network(x)

    # Hook to capture activations
    def get_activations(self, name):
        def hook(model, input, output):
            self.activations[name] = output.detach()
        return hook

# Function to add hooks and forward the input through the model
def forward_with_hooks(model, x):
    # Register hooks to capture activations
    for idx, layer in enumerate(model.network):
        if isinstance(layer, nn.Linear) or isinstance(layer, nn.ReLU):
            layer.register_forward_hook(model.get_activations(f"layer_{idx}"))

    # Forward pass through the model
    output = model(x)
    return output

# Sample usage
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DQN(device)
model.to(device)

# Example input state (batch size 1, 9 features for Tic-Tac-Toe board)
x = torch.rand(1, 9).to(device)

# Perform a forward pass with hooks capturing activations
output = forward_with_hooks(model, x)

# Visualize the model
dot = make_dot(output, params=dict(model.named_parameters()), show_attrs=True, show_saved=True)

# Save the detailed computational graph
dot.render('dqn_model_graph', format='png')

# Print the captured activations for better understanding
for name, activation in model.activations.items():
    print(f"Activation from {name}: {activation}")
