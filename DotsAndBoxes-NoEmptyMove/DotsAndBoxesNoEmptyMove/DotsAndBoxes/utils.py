class dotdict(dict):
    def __getattr__(self, name):
        return self[name]

    def __getstate__(self):
        return {
        'numIters': self['numIters'],             #訓練代數
        'numEps': self['numEps'],              # Number of complete self-play games to simulate during a new iteration.
        'tempThreshold': self['tempThreshold'],        # n步數以後只搜尋勝率最高的走步
        'updateThreshold': self['updateThreshold'],     # 新網路的勝率門檻 During arena playoff, new neural net will be accepted if threshold or more of games are won.
        'maxlenOfQueue': self['maxlenOfQueue'],    # Number of game examples to train the neural networks.
        'numMCTSSims': self['numMCTSSims'],          # Number of games moves for MCTS to simulate.
        'arenaCompare': self['arenaCompare'],         # Number of games to play during arena play to determine if new net will be accepted.
        'cpuct': self['cpuct'],

        'checkpoint': self['checkpoint'],
        'load_model': self['load_model'],
        'load_folder_file': self['load_folder_file'],
        'numItersForTrainExamplesHistory': self['numItersForTrainExamplesHistory'], #保留n代的訓練資料
    }

    def __setstate__(self,d):
        self.__dict__ = d 