import numpy as np

class TicTacToe:
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.board = np.zeros(9, dtype=int)  # 0: empty, 1: AI, -1: Opponent/Human
        self.done = False
        self.winner = None
        return self.board.copy()
    
    def available_actions(self):
        return [i for i in range(9) if self.board[i] == 0]
    
    def step(self, action, player):
        if self.done:
            raise Exception("Game is over")
        if self.board[action] != 0:
            raise Exception("Invalid move")
        self.board[action] = player
        reward = self.check_winner(player)
        self.done = reward != 0 or not self.available_actions()
        return self.board.copy(), reward, self.done
    
    def check_winner(self, player):
        combos = [
            (0, 1, 2), (3, 4, 5), (6, 7, 8),  # rows
            (0, 3, 6), (1, 4, 7), (2, 5, 8),  # columns
            (0, 4, 8), (2, 4, 6)              # diagonals
        ]
        for combo in combos:
            if all(self.board[i] == player for i in combo):
                self.winner = player
                return 1 if player == 1 else -1
        return 0
