import numpy as np

class RandomPlayer():
    def __init__(self, game):
        self.game = game

    def play(self, board):
        a = np.random.randint(self.game.getActionSize())
        valids = self.game.getValidMoves(board, 1)
        while valids[a]!=1:
            a = np.random.randint(self.game.getActionSize())
        return a

class HumanPlayer():
    def __init__(self, game):
        self.game = game

    def play(self, board):
        # display(board)
        valid = self.game.getValidMoves(board, 1)
        for i in range(len(valid)):
            if valid[i]:
                print("[", int(i/self.game.enlargeN), int(i%self.game.enlargeN), end="] ")
        
        while True:
            input_move = input()
            input_a = input_move.split(" ")

            if len(input_a) == 2:
                try:
                    x,y = [int(i) for i in input_a]
                    if ((1 <= x) and (x < self.game.enlargeN-1) and (1 <= y) and (y < self.game.enlargeN-1)) or \
                            ((x == self.game.enlargeN) and (y == 0)):
                        a = self.game.enlargeN * x + y if x != -1 else self.game.enlargeN ** 2
                        if valid[a]:
                            break
                except ValueError:
                    # Input needs to be an integer
                    'Invalid integer'
            print('Invalid move')
        return a