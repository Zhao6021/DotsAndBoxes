from Coach import Coach
from DotsAndBoxes.DotsAndBoxesGame import DotsAndBoxesGame as Game
from DotsAndBoxes.keras.NNet import NNetWrapper as nn  #可以改keras 變成其他的開源學習庫
from utils import *

#可以使用dot的dict
args = dotdict({
    'boardSize': 5,
    'numIters': 1,             #訓練代數
    'numEps': 1,              # Number of complete self-play games to simulate during a new iteration.
    'tempThreshold': 40,        # n步數以後只搜尋勝率最高的走步
    'updateThreshold': 0.6,     # 新網路的勝率門檻 During arena playoff, new neural net will be accepted if threshold or more of games are won.
    'maxlenOfQueue': 20000,    # Number of game examples to train the neural networks.
    'numMCTSSims': 400,          # Number of games moves for MCTS to simulate.
    'arenaCompare': 40,         # Number of games to play during arena play to determine if new net will be accepted.
    'cpuct': 1,

    'checkpoint': './temp/',
    'load_model': False,
    'load_folder_file': ('./temp/','best.pth.tar'),
    'numItersForTrainExamplesHistory': 20, #保留n代的訓練資料

})


if __name__ == "__main__":  #__name__ 是前檔案名 當檔案被直接運行時檔案名
    #初始化 創造一個新的Game並且定義他的nnet
    g = Game(args.boardSize)  #傳入DotsAndBoxesGame中-> n = 3 n是盤面大小
    nnet = nn(g) #把 g 傳入 othello.keras.NNet 的class NNetWrapper中

    if args.load_model:  #應該是可以讀入設定好的model
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])

    c = Coach(g, nnet, args) #將 game, nnet, args傳到 coach中 做初始化
    '''
    if args.load_model:  #可以load訓練好的範例?
        print("Load trainExamples from file")
        c.loadTrainExamples()
    '''
    c.learn() #開始訓練
