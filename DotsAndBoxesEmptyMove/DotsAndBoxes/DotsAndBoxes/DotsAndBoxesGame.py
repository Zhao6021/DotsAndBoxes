from __future__ import print_function
import sys
sys.path.append('..')
from Game import Game
from .DotsAndBoxesLogic import Board
import numpy as np

class DotsAndBoxesGame(Game):  #繼承class Game
    square_content = {
        -1: "B",
        +0: "-",
        +1: "A"
    }

    @staticmethod
    def getSquarePiece(piece):
        return DotsAndBoxesGame.square_content[piece]

    def __init__(self,n):
        self.n = n
        self.enlargeN = self.n + (self.n-1) + 4 # 第一排橫線的數量 + 每兩個橫線中的間格數 + 左右各兩個
    
    def getInitBoard(self):
        b = Board(self.n)
        return np.array(b.pieces)

    def getBoardSize(self):
        return (self.enlargeN,self.enlargeN)

    def getActionSize(self):
        return self.enlargeN * (self.enlargeN)  + 1 #橫線和直線各有n*(n+1)條 + 對手連下

        #return self.enlargeN * (self.enlargeN)  #橫線和直線各有n*(n+1)條 + 對手連下

    def getNextState(self, board, player, action):
        # 如果action 是 enlarge^2+1
        '''
        if action == self.enlargeN*self.enlargeN: #action 總數為 enlageN^2+1 此為最後一個也就是不走步
            return (board, -player)
        '''
        
        b = Board(self.n)
        b.pieces = np.copy(board)
        move = (int(action/self.enlargeN), action%self.enlargeN) #假設 enlargeN=9 action=0~ (9x9) 當 action=12時 則坐落在 (1,3)的位子
        b.execute_move(move, player)
        
        return (b.pieces, -player)



    def getValidMoves(self, board, player):
        valids = [0]*self.getActionSize()
        
        # 如果action 是 enlarge^2+1
        if board[0][0] == 6 or board[0][0] == -6:
            valids[-1] = 1   #把最後一個變成1 也就是不走步
            #board[0][0] = board[0][1]
            return np.array(valids)
        

        for i in range(0,self.enlargeN):
            for l in range(0,self.enlargeN):
                if board[i][l] == 0:
                    valids[i*self.enlargeN + l] = 1
        
        return np.array(valids)



    def getGameEnded(self, board, player):
        # return 0 if not ended, 1 if player 1 won, -1 if player 1 lost
        # player = 1
        b = Board(self.n)
        b.pieces = np.copy(board)
        if b.has_legal_moves(player):
            return 0
        if b.countDiff(player) > 0:
            return 1
        return -1

    def getCanonicalForm(self, board, player):
        return player*board

    def getSymmetries(self, board, pi):
        # 如果action 是 enlarge^2+1
        assert(len(pi) == self.getActionSize())  # 1 for pass
        pi_board = np.reshape(pi[:-1], self.getBoardSize())
        l = []

        for i in range(1, 5):
            for j in [True, False]:
                newB = np.rot90(board, i)  #逆時鐘旋轉 i*90度
                newPi = np.rot90(pi_board, i)
                if j:
                    newB = np.fliplr(newB)  #左右對調
                    newPi = np.fliplr(newPi)
                l += [(newB, list(newPi.ravel()) + [pi[-1]])] #ravel 跟 flatten很像 只是flatten會複製一份新的 ravel是同一份
        return l

    def stringRepresentation(self, board):
        return board.tostring()

    def stringRepresentationReadable(self, board):
        board_s = "".join(self.square_content[square] for row in board for square in row)
        return board_s

    def getScore(self, board, player):
        b = Board(self.enlargeN)
        b.pieces = np.copy(board)
        return b.countDiff(player)

    @staticmethod
    def display(board):
        n = board.shape[0] -1
        for y in range(1,n):
            print(y, end=" ")
        print("")
        print("-----------------------")
        for y in range(1,n):
            print(" ")
            for x in range(1,n):
                piece = board[y][x]    # get the piece to print
                if piece == 5:
                    print("o", end=" ")
                    continue
                if piece == 8 or piece == 0:
                    print(" ", end=" ")
                    continue
                if piece == 9 or piece == 7 or piece ==-9 or piece == -7:
                    if piece>0:
                        piece -= 8
                    else:
                        piece += 8
                    print(DotsAndBoxesGame.square_content[piece], end=" ")
                    continue
                if piece == -1:
                    piece = 2
                print (piece, end =" ")
            print(" ",y)


        print("-----------------------")
