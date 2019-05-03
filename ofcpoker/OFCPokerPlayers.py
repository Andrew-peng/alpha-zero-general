import numpy as np
from .OFCPokerLogic import OFCEvaluator


class RandomPlayer():
    def __init__(self, game):
        self.game = game

    def play(self, board):
        a = np.random.randint(self.game.getActionSize())
        valids = self.game.getValidMoves(board, 1)
        while valids[a]!=1:
            a = np.random.randint(self.game.getActionSize())
        return a


class HumanOFCPokerPlayer():
    def __init__(self, game):
        self.game = game

    def play(self, board):
        # display(board)
        valid = self.game.getValidMoves(board, 1)
        valid_moves = [i for i, v in enumerate(valid) if v]
        print("Valid moves:", valid_moves)

        while True:
            a = input()
            a = int(a)
            if a in (0, 1, 2) and valid[a]:
                break
            else:
                print('Invalid move')

        return a


# class GreedyOFCPokerPlayer():
#     def __init__(self, game):
#         self.game = game
#         self.evaluator = OFCEvaluator()

#     def play(self, board):
#         valids = self.game.getValidMoves(board, 1)
#         candidates = [x for x]
#         for a in range(self.game.getActionSize()):
#             if valids[a]==0:
#                 continue
#             nextBoard, _ = self.game.getNextState(board, 1, a)
#             score = self.game.getScore(nextBoard, 1)
#             candidates += [(-score, a)]
#         candidates.sort()
#         return candidates[0][1]
