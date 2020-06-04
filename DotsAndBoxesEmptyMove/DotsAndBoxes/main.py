from Coach import Coach
from DotsAndBoxes.DotsAndBoxesGame import DotsAndBoxesGame as Game
from DotsAndBoxes.keras.NNet import NNetWrapper as nn  #可以改keras 變成其他的開源學習庫
from utils import *

#可以使用dot的dict
args = dotdict({
    'numIters': 10,
    'numEps': 10,              # Number of complete self-play games to simulate during a new iteration.
    'tempThreshold': 15,        #
    'updateThreshold': 0.6,     # 新網路的勝率門檻 During arena playoff, new neural net will be accepted if threshold or more of games are won.
    'maxlenOfQueue': 20,    # Number of game examples to train the neural networks.
    'numMCTSSims': 10,          # Number of games moves for MCTS to simulate.
    'arenaCompare': 20,         # Number of games to play during arena play to determine if new net will be accepted.
    'cpuct': 1,

    'checkpoint': './temp/',
    'load_model': False,
    'load_folder_file': ('/dev/models/8x100x50','best.pth.tar'),
    'numItersForTrainExamplesHistory': 20,

})


if __name__ == "__main__":  #__name__ 是前檔案名 當檔案被直接運行時檔案名
    #初始化 創造一個新的Game並且定義他的nnet
    g = Game(3)  #傳入OthelloGame中-> n = 6 n是盤面大小
    nnet = nn(g) #把 g 傳入 othello.keras.NNet 的class NNetWrapper中

    if args.load_model:  #應該是可以讀入設定好的model
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])

    c = Coach(g, nnet, args) #將 game nnet args傳到 coach中 做初始化
    if args.load_model:  #可以load訓練好的範例?
        print("Load trainExamples from file")
        c.loadTrainExamples()
    c.learn() #開始訓練
