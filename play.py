# play.py
import torch
from game import TicTacToe

def print_board(board):
    symbols = {1: 'X', -1: 'O', 0: ' '}
    board_symbols = [symbols[x] for x in board]
    print(f"{board_symbols[0]} | {board_symbols[1]} | {board_symbols[2]}")
    print("---------")
    print(f"{board_symbols[3]} | {board_symbols[4]} | {board_symbols[5]}")
    print("---------")
    print(f"{board_symbols[6]} | {board_symbols[7]} | {board_symbols[8]}")
    print()

def play(model, device):
    model.to(device)  # Move the model to the device
    model.eval()      # Set model to evaluation mode
    game = TicTacToe()
    state = torch.FloatTensor(game.reset()).to(device)
    done = False

    print("Positions are numbered as follows:")
    print("0 | 1 | 2")
    print("---------")
    print("3 | 4 | 5")
    print("---------")
    print("6 | 7 | 8")
    print("\nYou are 'O' and the AI is 'X'. Good luck!\n")

    while not done:
        # AI's turn
        with torch.no_grad():
            q_values = model(state)
            masked_q_values = q_values.clone()
            for i in range(9):
                if i not in game.available_actions():
                    masked_q_values[i] = -float('inf')
            action = torch.argmax(masked_q_values).item()

        _, reward, done = game.step(action, 1)
        print_board(game.board)
        if done:
            if reward == 1:
                print("AI wins!")
            else:
                print("It's a tie!")
            break

        # Human's turn
        valid_move = False
        while not valid_move:
            try:
                user_action = int(input("Your move (0-8): "))
                if user_action in game.available_actions():
                    valid_move = True
                else:
                    print("Invalid move. Try again.")
            except:
                print("Please enter a number between 0 and 8.")
        _, reward, done = game.step(user_action, -1)
        state = torch.FloatTensor(game.board.copy()).to(device)
        print_board(game.board)
        if done:
            if reward == -1:
                print("You win!")
            else:
                print("It's a tie!")
            break
