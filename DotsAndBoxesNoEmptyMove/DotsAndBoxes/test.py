from DotsAndBoxes.DotsAndBoxesGame import DotsAndBoxesGame as Game

if __name__ == "__main__":
    g = Game(3)
    board = g.getInitBoard()
    action=0
    Game.display(board)
    player=int(1)
    while True:
        print("valids: ")
        valids = g.getValidMoves(board,player)
        for i in range(0,len(valids)):
            if valids[i]==1:
                print("[", int(i/g.enlargeN), int(i%g.enlargeN), end="] ")
        print("\ninput action  x y")
        actionXY=[int(x) for x in input().split()]
        if actionXY[0]==-1:
            break
        action = actionXY[0] * g.getBoardSize() [0] + actionXY[1]

        board = g.getCanonicalForm(board,-1)
        board,player = g.getNextState(board,-player,action)
        player = -player
        board = g.getCanonicalForm(board,-1)

        Game.display(board)
        end = g.getGameEnded(board, 1)
        if end != 0:
            print("winner=",end)
            break
    
    board = g.getCanonicalForm(board,-1)
    end = g.getGameEnded(board, 1)
    print("winner=",end)
    
    #Game.display(board)