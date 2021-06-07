import numpy as np
from pytorch_classification.utils import Bar, AverageMeter
import time
from datetime import datetime

class Arena_elo():
    """
    An Arena class where any 2 agents can be pit against each other.
    """
    def __init__(self, player1, player2, game, player1_name, player2_name, display=None):
        """
        Input:
            player 1,2: two functions that takes board as input, return action
            game: Game object
            display: a function that takes board as input and prints it (e.g.
                     display in othello/OthelloGame). Is necessary for verbose
                     mode.

        see othello/OthelloPlayers.py for an example. See pit.py for pitting
        human players/other baselines with each other.
        """
        self.player1 = player1
        self.player2 = player2
        self.game = game
        self.display = display

        self.R1 = 0 # player 1 score
        self.R2 = 363 # player 2 score

        self.player1_name = player1_name
        self.player2_name = player2_name

    def playGame(self, verbose=False):
        """
        Executes one episode of a game.

        Returns:
            either
                winner: player who won the game (1 if player1, -1 if player2)
            or
                draw result returned from the game that is neither 1, -1, nor 0.
        """
        players = [self.player2, None, self.player1]
        curPlayer = 1
        board = self.game.getInitBoard()
        it = 0
        while self.game.getGameEnded(board, curPlayer)==0:
            it+=1
            if verbose:
                assert(self.display)
                print("Turn ", str(it), "Player ", str(curPlayer))
                self.display(board)
            action = players[curPlayer+1](self.game.getCanonicalForm(board, curPlayer))

            valids = self.game.getValidMoves(self.game.getCanonicalForm(board, curPlayer),1)

            if valids[action]==0:
                print(action)
                assert valids[action] >0
            # print("Player ", str(curPlayer), 'take action ',action)
            board, curPlayer = self.game.getNextState(board, curPlayer, action)
        if verbose:
            assert(self.display)
            print("Game over: Turn ", str(it), "Result ", str(self.game.getGameEnded(board, 1)))
            self.display(board)

        return curPlayer*self.game.getGameEnded(board, curPlayer)

    def playGames(self, num, verbose=False):
        """
        Plays num games in which player1 starts num/2 games and player2 starts
        num/2 games.

        Returns:
            oneWon: games won by player1
            twoWon: games won by player2
            draws:  games won by nobody
        """
        eps_time = AverageMeter()
        bar = Bar('Arena.playGames', max=num)
        end = time.time()
        eps = 0
        maxeps = int(num)

        num = int(num/2)
        oneWon = 0
        twoWon = 0
        draws = 0
        for _ in range(num):
            gameResult = self.playGame(verbose=verbose)
            if gameResult==1:
                oneWon+=1
            elif gameResult==-1:
                twoWon+=1
            else:
                draws+=1
            
            #計算elo
            self.elo(gameResult)
            # bookkeeping + plot progress
            eps += 1
            #pgn log
            self.saveToPGN(gameResult,eps)

            eps_time.update(time.time() - end)
            end = time.time()
            bar.suffix  = '({eps}/{maxeps}) Eps Time: {et:.3f}s | Total: {total:} | ETA: {eta:}| Win: {one}:{two}'.format(eps=eps, maxeps=maxeps, et=eps_time.avg,
                                                                                                       total=bar.elapsed_td, eta=bar.eta_td , one=oneWon ,two = twoWon)
            bar.next()

        print('half')
        print(oneWon, twoWon, draws)
        print('elo: player1: ',self.R1,'player2',self.R2)

        self.player1, self.player2 = self.player2, self.player1
        
        for _ in range(num):
            gameResult = self.playGame(verbose=verbose)
            if gameResult==-1:
                oneWon+=1                
            elif gameResult==1:
                twoWon+=1
            else:
                draws+=1
            
            #計算elo
            self.elo(gameResult * -1) 
            # bookkeeping + plot progress
            eps += 1
            #pgn log
            self.saveToPGN(gameResult * -1,eps)

            eps_time.update(time.time() - end)
            end = time.time()
            bar.suffix  = '({eps}/{maxeps}) Eps Time: {et:.3f}s | Total: {total:} | ETA: {eta:}| Win: {one}:{two}'.format(eps=eps, maxeps=maxeps, et=eps_time.avg,
                                                                                                       total=bar.elapsed_td, eta=bar.eta_td , one=oneWon ,two = twoWon)
            bar.next()
            
        bar.finish()
        print('elo: player1: ',self.R1,'player2',self.R2)

        file_name = self.player1_name + '_vs_' + self.player2_name + '.pgn'
        self.saveToScript(file_name)

        return oneWon, twoWon, draws

    def saveToPGN(self, result , eps):
        file_name = './log/' + self.player1_name + '_vs_' + self.player2_name + '.pgn'
        f = open(file_name, 'a')
        f.write('[Event "' + self.player1_name + '_vs_' + self.player2_name + '"]\n')
        f.write('[Site "Taipei, TPE TPE"]\n')
        f.write('[Date "' + datetime.today().strftime('%Y.%m.%d') + '"]\n')
        f.write('[Round "' + str(eps) + '"]\n')
        f.write('[White "' + self.player1_name + '"]\n')
        f.write('[Black "' + self.player2_name + '"]\n')

        str_result = ''
        if result == 1:
            str_result = '1-0'
        elif result == -1:
            str_result = '0-1'
        else:
            str_result = '1/2-1/2'

        f.write('[Result "' + str_result + '"]\n')
        f.write(str_result+'\n\n')
        f.close()

    def saveToScript(self, file_name):
        script_name = './log/script.txt'
        f = open(script_name, 'a')
        f.write('readpgn '+file_name + '\n')
        f.close()


    def elo(self, result):
        E1 = self.computeScore(self.R2, self.R1)  #player 1 的勝負值
        E2 = self.computeScore(self.R1, self.R2)  #player 2 的勝負值

        score_adjust1 = 0
        score_adjust2 = 0
        K = self.computeK(self.R1) if self.computeK(self.R1) > self.computeK(self.R2)  else self.computeK(self.R2) #K為一常數
        if result == 1:
            score_adjust1 = 1
            score_adjust2 = 0
        elif result == -1:
            score_adjust1 = 0
            score_adjust2 = 1
        else:
            score_adjust1 = 0.5
            score_adjust2 = 0.5
        
        self.R1 = round (self.R1 + K * (score_adjust1 - E1) , 0)
        self.R2 = round (self.R2 + K * (score_adjust2 - E2) , 0)

    def computeK(self, rating):
        if rating >= 2400:
            return 16
        elif rating >= 2100:
            return 24
        else:
            return 32
    
    def computeScore(self, rating1, rating2):
        return 1 / (1+pow(10, (rating1 - rating2) / 400))