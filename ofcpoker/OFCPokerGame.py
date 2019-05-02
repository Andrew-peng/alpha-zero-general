from __future__ import print_function
import sys
import copy
from itertools import chain, combinations
sys.path.append('..')
from Game import Game
from .OFCPokerLogic import OFCPokerBoard
import numpy as np


class OFCPokerGame(Game):
    """
    This class specifies the base Game class. To define your own game, subclass
    this class and implement the functions below. This works when the game is
    two-player, adversarial and turn-based.

    Use 1 for player1 and -1 for player2.

    See othello/OthelloGame.py for an example implementation.
    """
    def __init__(self):
        pass

    def getInitBoard(self):
        """
        Returns:
            startBoard: a representation of the board (ideally this is the form
                        that will be the input to your neural network)
        """
        board = OFCPokerBoard()
        return board

    def getBoardSize(self):
        """
        Returns:
            (x,y): a tuple of board dimensions
        """
        # represent the poker board as 13 x 1 vector
        return (27, 1)

    def getActionSize(self):
        """
        Returns:
            actionSize: number of all possible actions
        """
        return 3

    def getNextState(self, board, player, action):
        """
        Input:
            board: current board
            player: current player (1 or -1)
            action: action taken by current player

        Returns:
            nextBoard: board after applying action
            nextPlayer: player who plays in the next turn (should be -player)
        """
        board.execute_move(action, player)
        player = -1 * player
        return board, player

    def getValidMoves(self, board, player):
        """
        Input:
            board: current board
            player: current player

        Returns:
            validMoves: a binary vector of length self.getActionSize(), 1 for
                        moves that are valid from the current board and player,
                        0 for invalid moves
        """
        return np.array(board.get_legal_moves(player))

    def getGameEnded(self, board, player):
        """
        Input:
            board: current board
            player: current player (1 or -1)

        Returns:
            r: 0 if game has not ended. 1 if player won, -1 if player lost,
               small non-zero value for draw.
               
        """
        if not board.game_ended():
            return 0
        else:
            return board.get_score(player)

    def getCanonicalForm(self, board, player):
        """
        Input:
            board: current board
            player: current player (1 or -1)

        Returns:
            canonicalBoard: returns canonical form of board. The canonical form
                            should be independent of player. For e.g. in chess,
                            the canonical form can be chosen to be from the pov
                            of white. When the player is white, we can return
                            board as is. When the player is black, we can invert
                            the colors and return the board.
        """
        board = copy.deepcopy(board)
        if player == -1:
            board_p, card_p = board.boards[player], board.current_cards[player]
            board.boards[player] = board.boards[-player]
            board.current_cards[player] = board.current_cards[-player]
            board.boards[-player] = board_p
            board.current_cards[-player] = card_p
        return board

    def getSymmetries(self, board, pi):
        """
        Input:
            board: current board
            pi: policy vector of size self.getActionSize()

        Returns:
            symmForms: a list of [(board,pi)] where each tuple is a symmetrical
                       form of the board and the corresponding pi vector. This
                       is used when training the neural network from examples.
        """
        # TODO: Just return the one board
        # Current player is p1
        # Checking my board for permutable streets
        # my_street_lens = board.get_permutable_streets(1)
        # permutable = [(1, i) for i in range(len(my_street_lens)) if my_street_lens[i]]
        # # Checking opp board for permutable streets
        # opp_street_lens = board.get_permutable_streets(-1)
        # permutable.extend([(-1, i) for i in range(len(opp_street_lens)) if my_street_lens[i]])
        # # Generate all subsets

        # def generate_subsets(l):
        #     return chain.from_iterable(combinations(l,n) for n in range(len(l)+1))

        # street_permutations = generate_subsets(permutable)
        symmetries = [(board, pi)]
        # for to_permute in street_permutations:
        #     b = copy.deepcopy(board)
        #     p = copy.deepcopy(pi)
        #     for player, street in to_permute:
        #         b.permute(street, player)
        #     symmetries.append((b, p))
        return symmetries

    def stringRepresentation(self, board):
        """
        Input:
            board: current board

        Returns:
            boardString: a quick conversion of board to a string format.
                         Required by MCTS for hashing.
        """
        return repr(board)

