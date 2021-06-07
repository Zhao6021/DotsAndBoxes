import argparse
import os
import shutil
import time
import random
import numpy as np
import math
import sys
sys.path.append('../..')
from utils import *
from NeuralNet import NeuralNet

import argparse

from .DotsAndBoxesNNet import DotsAndBoxesNNet as onnet

#可以使用dot的dictionary
args = dotdict({  
    'lr': 0.001,
    'dropout': 0.3,
    'epochs': 10,
    'batch_size': 64,
    'cuda': True,
    'num_channels': 512,
})

class NNetWrapper(NeuralNet):
    def __init__(self, game):
        self.nnet = onnet(game, args)   #把遊戲和args傳進DotsAndBoxesNNet裡
        self.board_x, self.board_y = game.getBoardSize()   #取得盤面大小並初始化
        self.action_size = game.getActionSize()            #取得走步種數
        self.game = game

    def train(self, examples):
        """
        examples: list of examples, each example is of form (board, pi, v)
        """
        input_boards, target_pis, target_vs = list(zip(*examples))
        input_boards = np.asarray(input_boards)
        # print(target_pis)
        #decode
        input_decode_boards = []
        input_decode_p = []
        for i in range(0,len(input_boards)):
            input_decode_boards.append( self.game.boardDecode(input_boards[i]))
            input_decode_p.append(self.pDecode(target_pis[i]))
        input_decode_boards = np.array(input_decode_boards , dtype = 'int8')
        # input_decode_p = np.array(input_decode_p , dtype = 'float32')
        
        target_pis = np.asarray(input_decode_p)
        target_vs = np.asarray(target_vs)
        self.nnet.model.fit(x = input_decode_boards, y = [target_pis, target_vs], batch_size = args.batch_size, epochs = args.epochs)

    def predict(self, board):
        """
        board: np array with board
        """
        # timing
        start = time.time()

        # preparing input
        board = board[np.newaxis, :, :]

        # run
        pi, v = self.nnet.model.predict(board)

        #print('PREDICTION TIME TAKEN : {0:03f}'.format(time.time()-start))
        return pi[0], v[0]

    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print("Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.mkdir(folder)
        else:
            print("Checkpoint Directory exists! ")
        self.nnet.model.save_weights(filepath)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        # https://github.com/pytorch/examples/blob/master/imagenet/main.py#L98
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise("No model in path {}".format(filepath))
        self.nnet.model.load_weights(filepath)

    def pEncode(self,p):
        index = []
        value = []
        index.append(len(p))
        value.append(0)
        for i in range(len(p)):
            if p[i] != 0:
                index.append(i)
                value.append( float(p[i]) )
        index = np.array(index, dtype = "uint8")
        value = np.array(value, dtype = "float32")
        re = []
        re.append(index)
        re.append(value)
        re = np.array(re , dtype = object)
        return re

    def pDecode(slef,p):
        re = [0.0 for i in range(p[0][0])]
        for i in range(1,len(p[0])):
            re[p[0][i]] = p[1][i]

        return re 
