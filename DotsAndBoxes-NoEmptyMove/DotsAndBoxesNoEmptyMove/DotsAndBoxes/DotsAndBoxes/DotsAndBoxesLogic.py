'''
x is the row, y is the column.
'''
class Board():

    directions = [(0,1),(1,0),(0,-1),(-1,0)]

    def __init__(self, n):
        "Set up initial board configuration."
        self.n = n
        # Create the empty board array.
        self.enlargeN = self.n + (self.n-1) + 4 # 3 + 2 + 4= 9
        self.pieces = [None]*self.enlargeN
        for i in range(0,self.enlargeN):
            self.pieces[i] = [0]*self.enlargeN
        
        #設定邊界.....
        for i in range (0,self.enlargeN):
            if i==0 or i==self.enlargeN-1 : #5 5 5 5 5 5 5 5 5
                for l in range(0,self.enlargeN):
                    self.pieces[i][l]=5
                continue
            if i % 2==1: #5 5 0 5 0 5 0 5 5 橫線
                self.pieces[i][0] = 5
                self.pieces[i][self.enlargeN-1] = 5 
                for l in range(1,self.enlargeN-1):
                    if l%2==0:
                        self.pieces[i][l]=0
                    else:
                        self.pieces[i][l]=5
            else: #5 0 8 0 8 0 8 0 5 直線 8為格子
                self.pieces[i][0] = 5
                self.pieces[i][self.enlargeN-1] = 5 
                for l in range(1,self.enlargeN-1):
                    if l%2==0:
                        self.pieces[i][l]=8
                    else:
                        self.pieces[i][l]=0
    
    def got_point(self,x,y): #判斷是否得分
        for i in self.directions:
            if self.pieces[x+i[0]][y+i[1]]==0:
                return False
        return True

    def execute_move(self, move, color):
        self.pieces[move[0]][move[1]]=color
        got=False
        for i in self.directions:
            tmpGot = False
            if self.pieces[move[0]+i[0]][move[1]+i[1]] == 8 or self.pieces[move[0]+i[0]][move[1]+i[1]] == -8: #搜尋點的上下左右是否有8
                tmpGot=self.got_point(move[0]+i[0],move[1]+i[1])
                if tmpGot==True:
                    self.pieces[move[0]+i[0]][move[1]+i[1]]+=color #得分， 8+color
                    got = True
        return got
    
    def has_legal_moves(self, color):
        for i in range(0,self.enlargeN):
            for l in range(0,self.enlargeN):
                if self.pieces[i][l] == 0:
                    return True
        return False

    def countDiff(self, color):
        """Counts the # pieces of the given color
        (1 for white, -1 for black, 0 for empty spaces)"""
        count = 0
        for y in range(2,self.n*2+1,2):
            for x in range(2,self.n*2+1,2):
                if self.pieces[x][y]<0:
                    count += self.pieces[x][y]+8
                else:
                    count += self.pieces[x][y]-8
        
        count *= color
        return count