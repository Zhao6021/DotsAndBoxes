from collections import deque
from Arena import Arena
from MCTS import MCTS
import numpy as np
from pytorch_classification.utils import Bar, AverageMeter
import time, os, sys
from pickle import Pickler, Unpickler
from random import shuffle

from multiprocessing import Pool
import gc
import multiprocessing as mp
from DotsAndBoxes.DotsAndBoxesGame import DotsAndBoxesGame as Game
from DotsAndBoxes.keras.NNet import NNetWrapper as nn  #可以改keras 變成其他的開源學習庫
from utils import *

class Coach():
    """
    This class executes the self-play + learning. It uses the functions defined
    in Game and NeuralNet. args are specified in main.py.
    """
    def __init__(self, game, nnet, args):
        self.game = game    
        self.nnet = nnet
        self.pnet = self.nnet.__class__(self.game)  # the competitor network  __class__會返回一個nnet的實例
        self.args = args  #來自main
        self.mcts = MCTS(self.game, self.nnet, self.args)  #初始化MCTS
        self.trainExamplesHistory = []    # history of examples from args.numItersForTrainExamplesHistory latest iterations
        self.skipFirstSelfPlay = False    # can be overriden in loadTrainExamples() 用來推翻 loadTrainExamples()這個函式
        self.multiprocessing = False
        # self.cpu = mp.cpu_count()
        self.cpu = 1.5
        self.maxtasksperchild = 1

    def executeEpisode(self):
        """
        This function executes one episode of self-play, starting with player 1.
        As the game is played, each turn is added as a training example to
        trainExamples. The game is played till the game ends. After the game
        ends, the outcome of the game is used to assign values to each example
        in trainExamples.

        It uses a temp=1 if episodeStep < tempThreshold, and thereafter
        uses temp=0.

        Returns:
            trainExamples: a list of examples of the form (canonicalBoard,pi,v)
                           pi is the MCTS informed policy vector, v is +1 if
                           the player eventually won the game, else -1.
        """
        trainExamples = []
        board = self.game.getInitBoard()
        self.curPlayer = 1
        episodeStep = 0
        while True:
            episodeStep += 1
            canonicalBoard = self.game.getCanonicalForm(board,self.curPlayer)
            temp = int(episodeStep < self.args.tempThreshold)
            pi = self.mcts.getActionProb(canonicalBoard, temp=temp)
            sym = self.game.getSymmetries(canonicalBoard, pi)
            count = 0
            for b,p in sym:
                
                boardEnccode = self.game.boardEncode(b,self.curPlayer)
                # boardEnccode = b
                pEncode = self.nnet.pEncode(p)
                # pEncode = p
                trainExamples.append([boardEnccode, self.curPlayer, pEncode, None])
                count+=1

            action = np.random.choice(len(pi), p=pi)
            board, self.curPlayer = self.game.getNextState(board, self.curPlayer, action)

            r = self.game.getGameEnded(board, self.curPlayer)

            if r!=0:
                return [(x[0],x[2],r*((-1)**(x[1]!=self.curPlayer))) for x in trainExamples]

    def learn(self):
        """
        Performs numIters iterations with numEps episodes of self-play in each
        iteration. After every iteration, it retrains neural network with
        examples in trainExamples (which has a maximum length of maxlenofQueue).
        It then pits the new neural network against the old one and accepts it
        only if it wins >= updateThreshold fraction of games.
        """

        for i in range(1, self.args.numIters+1):
            # bookkeeping
            print('------ITER ' + str(i) + '------')
            # examples of the iteration
            if not self.skipFirstSelfPlay or i>1:
                saveSelfPlayTimeLog('------ITER ' + str(i) + '------')
                iterationTrainExamples = deque([], maxlen=self.args.maxlenOfQueue)

                selfPlayStartTime = time.time()

                if self.multiprocessing:
                    pool = Pool(processes = self.cpu, maxtasksperchild = self.maxtasksperchild)
                    for eps in range(self.args.numEps):
                        self.mcts = MCTS(self.game, self.nnet, self.args)   # reset search tree
                        # re = pool.apply_async(selfPlay, args=(eps, self.game, self.args))
                        # iterationTrainExamples += re.get()
                        re = pool.starmap(selfPlay, [(str(eps), self.args)])
                        iterationTrainExamples += re[0]
                        gc.collect()

                    pool.close()
                    pool.join()
                else:
                    for eps in range(self.args.numEps):
                        selfStartTime = time.time()
                        
                        self.mcts = MCTS(self.game, self.nnet, self.args)   # reset search tree
                        re = self.executeEpisode()
                        iterationTrainExamples += re

                        print('Episode ',eps,' eps cost time = %.3f'%(time.time()-selfStartTime) ,' sec')
                        saveSelfPlayTimeLog('Episode ' + str(eps) + ' eps cost time = %.3f'%(time.time()-selfStartTime) + ' sec')

                print('SelfPlay total cost time = %.3f'%(time.time()-selfPlayStartTime),' sec')
                saveSelfPlayTimeLog('SelfPlay total cost time = %.3f'%(time.time()-selfPlayStartTime) + ' sec')
                
                # save the iteration examples to the history
                self.trainExamplesHistory.append(iterationTrainExamples)
                
            if len(self.trainExamplesHistory) > self.args.numItersForTrainExamplesHistory:
                print("len(trainExamplesHistory) =", len(self.trainExamplesHistory), " => remove the oldest trainExamples")
                self.trainExamplesHistory.pop(0)
            # backup history to a file
            # NB! the examples were collected using the model from the previous iteration, so (i-1)  
            saveStart = time.time()
            self.saveTrainExamples(i-1)
            saveEnd = time.time()
            self.saveTimeLog(i,saveEnd-saveStart)
            
            # shuffle examples before training
            trainExamples = []
            for e in self.trainExamplesHistory:
                trainExamples.extend(e)
            shuffle(trainExamples)

            # training new network, keeping a copy of the old one
            self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            self.pnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            pmcts = MCTS(self.game, self.pnet, self.args)
            
            self.nnet.train(trainExamples)
            nmcts = MCTS(self.game, self.nnet, self.args)

            print('PITTING AGAINST PREVIOUS VERSION')
            arena = Arena(lambda x: np.argmax(pmcts.getActionProb(x, temp=0)),
                          lambda x: np.argmax(nmcts.getActionProb(x, temp=0)), self.game)
            pwins, nwins, draws = arena.playGames(self.args.arenaCompare)

            print('NEW/PREV WINS : %d / %d ; DRAWS : %d' % (nwins, pwins, draws))
            if pwins+nwins == 0 or float(nwins)/(pwins+nwins) < self.args.updateThreshold:
                print('REJECTING NEW MODEL')
                self.nnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            else:
                print('ACCEPTING NEW MODEL')
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename=self.getCheckpointFile(i))
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='best.pth.tar')                

    def getCheckpointFile(self, iteration):
        return 'checkpoint_' + str(iteration) + '.pth.tar'

    def saveTrainExamples(self, iteration):
        folder = self.args.checkpoint
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(folder, self.getCheckpointFile(iteration)+".examples")
        with open(filename, "wb+") as f:
            Pickler(f).dump(self.trainExamplesHistory)
        f.closed

    def loadTrainExamples(self):
        modelFile = os.path.join(self.args.load_folder_file[0], self.args.load_folder_file[1])
        examplesFile = modelFile+".examples"
        if not os.path.isfile(examplesFile):
            print(examplesFile)
            r = input("File with trainExamples not found. Continue? [y|n]")
            if r != "y":
                sys.exit()
        else:
            print("File with trainExamples found. Read it.")
            with open(examplesFile, "rb") as f:
                self.trainExamplesHistory = Unpickler(f).load()
            f.closed
            # examples based on the model were already collected (loaded)
            self.skipFirstSelfPlay = True

    def saveTimeLog(self,iter,time):
        f = open('./log/save_time_log.txt','a')
        f.write( str(iter) +' iter save with time: '+ str(time/60) +' min\n')
        f.close

def selfPlay(eps,args):
    selfStartTime = time.time()

    g = Game(args.boardSize)

    nnet = nn(g)
    c = Coach(g, nnet, args)
    re = c.executeEpisode()
    print('Episode '+ eps +' eps cost time = %.3f'%(time.time()-selfStartTime) ,' sec')
    saveSelfPlayTimeLog('Episode ' + eps + ' eps cost time = %.3f'%(time.time()-selfStartTime) + ' sec')
    return re

def saveSelfPlayTimeLog(string):
        f = open('./log/self_play_time_log.txt','a')
        f.write(string+'\n')
        f.close