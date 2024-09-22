import torch
import os
from dotenv import load_dotenv
from trainer import Trainer
from model import DQN
from play import play

# Load environment variables
load_dotenv()

# Set device to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

if __name__ == "__main__":
    model_path = 'tic_tac_toe_model.pth'
    trainer = None  # Initialize trainer variable

    if os.path.exists(model_path):
        choice = input("A trained model exists. Do you want to retrain it? (y/n): ").strip().lower()
        if choice == 'y':
            # Load the existing model for further training
            print("Loading the existing model for further training...")
            policy_net = DQN(device)
            policy_net.load_state_dict(torch.load(model_path, map_location=device))
            trainer = Trainer(device, policy_net)  # Create an instance of Trainer
            policy_net = trainer.train()
        else:
            print("Loading the trained model for playing...")
            policy_net = DQN(device)
            policy_net.load_state_dict(torch.load(model_path, map_location=device))
            policy_net.eval()  # Set the model to evaluation mode
    else:
        # Train the model from scratch
        print("Training the model from scratch...")
        trainer = Trainer(device)  # Create an instance of Trainer
        policy_net = trainer.train()

    # Save the model after training
    if trainer:  # Check if trainer is defined
        trainer.save_model(model_path)

    # Play against the AI
    play(policy_net, device)
