import math
import numpy as np
EPS = 1e-8

class MCTS():
    """
    This class handles the MCTS tree.
    """

    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.args = args
        self.Qsa = {}       # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}       # stores #times edge s,a was visited
        self.Ns = {}        # stores #times board s was visited
        self.Ps = {}        # stores initial policy (returned by neural net)

        self.Es = {}        # stores game.getGameEnded ended for board s
        self.Vs = {}        # stores game.getValidMoves for board s

        self.labels = {}    # stores label of (s) 0:draw -1:lose 1:win
        self.unknown_children_count = {} # store unknown_children_count
        self.child_2_parent = {}    # store key:child value:parent,child player
        self.can_win_or_draw = {}   # 0:can_draw 1:can_win
        self.parent_2_child = {}    # store key:(parent , a) value:child , next_player

        self.got_winning_branch = False

    def getActionProb(self, canonicalBoard, temp=1):
        """
        This function performs numMCTSSims simulations of MCTS starting from
        canonicalBoard.

        Returns:
            probs: a policy vector where the probability of the ith action is
                   proportional to Nsa[(s,a)]**(1./temp)
        """
        s = self.game.stringRepresentation(canonicalBoard)
        self.ori_string_board = s
        for i in range(self.args.numMCTSSims):
            # print('node = ', i)
            v = self.search(canonicalBoard)
            self.got_winning_branch = False

        valid = self.game.getValidMoves(canonicalBoard,1)

        if s in self.labels and self.labels[s] == 1:
            win_action = []
            minA = float('inf')
            minAindex = 0
            for a in range(self.game.getActionSize()):
                if valid[a]:

                    if (s,a) in self.parent_2_child:
                        next_tmp_s = self.parent_2_child[ (s,a) ][0]
                        next_tmp_player = self.parent_2_child[ (s,a) ][1]
                    else:
                        next_tmp_s, next_tmp_player = self.game.getNextState(canonicalBoard, 1, a)
                        next_tmp_s = self.game.getCanonicalForm(next_tmp_s, next_tmp_player)
                        next_tmp_s = self.game.stringRepresentation(next_tmp_s)
                        self.parent_2_child[(s,a)] = [ next_tmp_s , next_tmp_player]

                    if next_tmp_s in self.labels:
                        if self.labels[next_tmp_s] == next_tmp_player : #下一個node label 是 1 且玩家也是1 or 下一個
                            if self.Ns[next_tmp_s] > 0 and self.Ns[next_tmp_s] < minA:
                                minAindex = a
                                minA = self.Ns[next_tmp_s]
                            win_action.append(self.Ns[next_tmp_s])
                            continue
                win_action.append(0)
            probs = [0]*len(win_action)
            probs[minAindex]=1
            return probs

        counts = [self.Nsa[(s,a)] if (s,a) in self.Nsa else 0 for a in range(self.game.getActionSize())]
        if sum(counts) == 0:
            counts = valid
        if temp==0:
            bestA = np.argmax(counts)
            probs = [0]*len(counts)
            probs[bestA]=1
            return probs

        counts = [x**(1./temp) for x in counts]
        counts_sum = float(sum(counts))
        probs = [x/counts_sum for x in counts]
        return probs


    def search(self, canonicalBoard):
        """
        This function performs one iteration of MCTS. It is recursively called
        till a leaf node is found. The action chosen at each node is one that
        has the maximum upper confidence bound as in the paper.

        Once a leaf node is found, the neural network is called to return an
        initial policy P and a value v for the state. This value is propagated
        up the search path. In case the leaf node is a terminal state, the
        outcome is propagated up the search path. The values of Ns, Nsa, Qsa are
        updated.

        NOTE: the return values are the negative of the value of the current
        state. This is done since v is in [-1,1] and if v is the value of a
        state for the current player, then its value is -v for the other player.

        Returns:
            v: the negative of the value of the current canonicalBoard
        """
        #print(canonicalBoard)
        s = self.game.stringRepresentation(canonicalBoard)
        if s not in self.Es:
            self.Es[s] = self.game.getGameEnded(canonicalBoard, 1)

            #exact_win initialize      
            self.unknown_children_count[s] = np.sum(self.game.getValidMoves(canonicalBoard, 1))
        
        if self.Es[s]!=0:
            # terminal node
            # exact_win set label
            self.Ns[s] = 1
            game_result = self.game.getGameWin(canonicalBoard, 1)
            self.mark(s,game_result)
            self.got_winning_branch = True
            return -self.Es[s]

        if s not in self.Ps:
            # leaf node
            self.Ps[s], v = self.nnet.predict(canonicalBoard)
            valids = self.game.getValidMoves(canonicalBoard, 1)
            self.Ps[s] = self.Ps[s]*valids      # masking invalid moves
            sum_Ps_s = np.sum(self.Ps[s])
            if sum_Ps_s > 0:
                self.Ps[s] /= sum_Ps_s    # renormalize
            else:
                # if all valid moves were masked make all valid moves equally probable
                
                # NB! All valid moves may be masked if either your NNet architecture is insufficient or you've get overfitting or something else.
                # If you have got dozens or hundreds of these messages you should pay attention to your NNet and/or training process.   
                print("All valid moves were masked, do workaround.")
                self.Ps[s] = self.Ps[s] + valids
                self.Ps[s] /= np.sum(self.Ps[s])

            self.Vs[s] = valids
            self.Ns[s] = 0
            return -v

        valids = self.Vs[s]
        cur_best = -float('inf')
        best_act = -1
        got_win_action = -1
        count = 0
        # pick the action with the highest upper confidence bound
        for a in range(self.game.getActionSize()):
            if valids[a]:

                # exact_win select phase
                
                if (s,a) in self.parent_2_child:
                    next_tmp_s = self.parent_2_child[ (s,a) ][0]
                    next_tmp_player = self.parent_2_child[ (s,a) ][1]
                else:
                    next_tmp_s, next_tmp_player = self.game.getNextState(canonicalBoard, 1, a)
                    next_tmp_s = self.game.getCanonicalForm(next_tmp_s, next_tmp_player)
                    next_tmp_s = self.game.stringRepresentation(next_tmp_s)
                    self.parent_2_child[(s,a)] = [ next_tmp_s , next_tmp_player]

                if next_tmp_s in self.labels:
                    got_win = self.got_same_node(s,next_tmp_s,next_tmp_player)
                    if got_win:
                        got_win_action = a
                    continue

                count+=1
                if (s,a) in self.Qsa:
                    u = self.Qsa[(s,a)] + self.args.cpuct*self.Ps[s][a]*math.sqrt(self.Ns[s])/(1+self.Nsa[(s,a)])
                else:
                    u = self.args.cpuct*self.Ps[s][a]*math.sqrt(self.Ns[s] + EPS)     # Q = 0 ?

                if u > cur_best:
                    cur_best = u
                    best_act = a

        if s in self.unknown_children_count:
            self.unknown_children_count[s] = count

        a = best_act
        v = 0
        next_player = 0
        if a != -1 and got_win_action == -1:
            next_s, next_player = self.game.getNextState(canonicalBoard, 1, a)
            next_s = self.game.getCanonicalForm(next_s, next_player)

            #exact win update child_2_parent
            string_next_s = self.game.stringRepresentation(next_s)
            self.child_2_parent[string_next_s] = [s,next_player]

            v = self.search(next_s)
            if next_player == 1:
                v = -v

        # exact win backprobagation
        self.check_label(s)

        if got_win_action != -1:
            v = self.labels[s]
            a = got_win_action

        if (s,a) in self.Qsa:
            self.Qsa[(s,a)] = (self.Nsa[(s,a)]*self.Qsa[(s,a)] + v)/(self.Nsa[(s,a)]+1)
            self.Nsa[(s,a)] += 1

        else:
            self.Qsa[(s,a)] = v
            self.Nsa[(s,a)] = 1

        self.Ns[s] += 1

        return -v

    def check_label(self,string_board):
        if string_board in self.can_win_or_draw and self.can_win_or_draw[string_board] == 1:
            self.mark(string_board,1)
        elif self.unknown_children_count[string_board] <= 0:
            if string_board in self.can_win_or_draw and self.can_win_or_draw[string_board] == 0:
                self.mark(string_board,0)
            else:
                self.mark(string_board,-1)
    
    def got_same_node(self,parent_s, child_s, next_player):
        if parent_s not in self.can_win_or_draw or (parent_s in self.can_win_or_draw and self.can_win_or_draw[parent_s] != 1):
            if next_player == 1:
                if self.labels[child_s] == 1:
                    self.can_win_or_draw[parent_s] = 1
                    return True
                elif self.labels[child_s] == 0:
                    self.can_win_or_draw[parent_s] = 0
                return False
            else:
                if self.labels[child_s] == -1:
                    self.can_win_or_draw[parent_s] = 1
                    return True
                elif self.labels[child_s] == 0:
                    self.can_win_or_draw[parent_s] = 0
                return False
        return False

    def mark(self, string_board, label):
        if string_board == self.ori_string_board:
            self.labels[string_board] = label
        else:
            parent_string_board = self.child_2_parent[string_board][0]
            child_player = self.child_2_parent[string_board][1]
            self.unknown_children_count[parent_string_board] -= 1
            self.labels[string_board] = label
            if child_player == 1:
                if label == 1:
                    self.can_win_or_draw[parent_string_board] = 1
                elif label == 0:
                    self.can_win_or_draw[parent_string_board] = 0
            else:
                if label == -1:
                    self.can_win_or_draw[parent_string_board] = 1
                elif label == 0:
                    self.can_win_or_draw[parent_string_board] = 0

        