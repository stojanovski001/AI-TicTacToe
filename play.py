#!/usr/bin/env python3

import torch
import os
from game import TicTacToe

def clear_terminal():
    os.system('cls' if os.name == 'nt' else 'clear')

def print_board(board):
    symbols = {1: 'X', -1: 'O', 0: ' '}
    board_symbols = [symbols[x] for x in board]
    print(f"{board_symbols[0]} | {board_symbols[1]} | {board_symbols[2]}")
    print("---------")
    print(f"{board_symbols[3]} | {board_symbols[4]} | {board_symbols[5]}")
    print("---------")
    print(f"{board_symbols[6]} | {board_symbols[7]} | {board_symbols[8]}")
    print()

def print_empty_board():
    clear_terminal()  # Clear the terminal before printing
    print(" 1 | 2 | 3")
    print("-----------")
    print(" 4 | 5 | 6")
    print("-----------")
    print(" 7 | 8 | 9")
    print()

def play(model, device):
    model.to(device)  # Move the model to the device
    model.eval()      # Set model to evaluation mode
    game = TicTacToe()
    state = torch.FloatTensor(game.reset()).to(device)
    done = False

    # Ask user who goes first
    while True:
        first_move = input("Do you want to go first? (y/n): ").strip().lower()
        if first_move in ['y', 'n']:
            break
        print("Please enter 'y' for yes or 'n' for no.")

    if first_move == 'y':
        print("\nYou are 'X' and the AI is 'O'. Good luck!\n")
    else:
        print("\nThe AI will go first. You are 'X' and the AI is 'O'. Good luck!\n")
        with torch.no_grad():
            q_values = model(state)
            masked_q_values = q_values.clone()
            for i in range(9):
                if i not in game.available_actions():
                    masked_q_values[i] = -float('inf')
            action = torch.argmax(masked_q_values).item()
        _, reward, done = game.step(action, -1)  # AI plays as 'O'
        print_empty_board()
        print_board(game.board)
        if done:
            if reward == -1:
                print("You win!")
            else:
                print("It's a tie!")
            return

    # print_empty_board()

    while not done:
        # Human's turn
        valid_move = False
        while not valid_move:
            try:
                user_action = int(input("Your move (1-9): ")) - 1  # Adjust input to match 0-8
                if user_action in game.available_actions():
                    valid_move = True
                else:
                    print("Invalid move. Try again.")
            except ValueError:
                print("Please enter a number between 1 and 9.")
        _, reward, done = game.step(user_action, 1)  # Human plays as 'X'
        state = torch.FloatTensor(game.board.copy()).to(device)
        print_board(game.board)
        if done:
            if reward == 1:
                print("You win!")
            else:
                print("It's a tie!")
            break

        # AI's turn
        with torch.no_grad():
            q_values = model(state)
            masked_q_values = q_values.clone()
            for i in range(9):
                if i not in game.available_actions():
                    masked_q_values[i] = -float('inf')
            action = torch.argmax(masked_q_values).item()

        _, reward, done = game.step(action, -1)  # AI plays as 'O'
        print_empty_board()
        print_board(game.board)
        if done:
            if reward == -1:
                print("AI wins!")
            else:
                print("It's a tie!")
            break

if __name__ == "__main__":
    # Here you would typically load your model
    pass  # Add your model loading code here if needed
