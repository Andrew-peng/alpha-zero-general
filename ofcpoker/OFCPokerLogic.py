import pickle
import random

import torch
import tensorflow as tf
import numpy as np

from deuces import Card, Deck, Evaluator
from termcolor import colored


CARDS = [
    '_', '2s', '3s', '4s', '5s', '6s', '7s', '8s', '9s', 'Ts', 'Js', 'Qs', 'Ks', 'As',
    '2h', '3h', '4h', '5h', '6h', '7h', '8h', '9h', 'Th', 'Jh', 'Qh', 'Kh', 'Ah',
    '2d', '3d', '4d', '5d', '6d', '7d', '8d', '9d', 'Td', 'Jd', 'Qd', 'Kd', 'Ad',
    '2c', '3c', '4c', '5c', '6c', '7c', '8c', '9c', 'Tc', 'Jc', 'Qc', 'Kc', 'Ac']

# HACK
def pretty_suits(cards):
    for i in range(1, len(cards)):
        if cards[i][1] == 's':
            cards[i] = cards[i][0] + " " + u"\u2660".encode('utf-8') # spades
        elif cards[i][1] == 'h':
            cards[i] = cards[i][0] + " " + colored(u"\u2764".encode('utf-8'), 'red') # hearts
        elif cards[i][1] == 'd':
            cards[i] = cards[i][0] + " " + colored(u"\u2666".encode('utf-8'), 'red') # diamonds
        else:
            cards[i] = cards[i][0] + " " + u"\u2663".encode('utf-8') # clubs
    return cards

CARD_MAP = dict(zip(pretty_suits(CARDS), list(range(53))))


FRONT_LOOKUP = pickle.load(open("ofcpoker/front_lookup.p", 'rb'))


def card_to_int(card):
    return CARD_MAP[Card.int_to_pretty_str(card)[3:-3]]


def to_int53(street, length):
    """
    Converts list of deuces cards to list of integers, padded to length with 0
    """
    assert length in (3, 5)
    cur_street = [card_to_int(i) for i in street]
    while len(cur_street) < length:
        cur_street.append(0)
    return cur_street


class OFCEvaluator(Evaluator):
    """deuces' evaluator class extended to score an OFC Front."""
    def __init__(self):
        super(OFCEvaluator, self).__init__()

        self.hand_size_map = {
            3: self._three,
            5: self._five,
            6: self._six,
            7: self._seven
        }

    def _three(self, cards):
        prime = Card.prime_product_from_hand(cards)
        return FRONT_LOOKUP[prime]


class OFCBoard(object):
    """Represent the three streets of an OFC game for one player."""

    def __init__(self):
        self.clear()

    def clear(self):
        self.front = []
        self.mid = []
        self.back = []

    def pretty(self):
        print('Front:')
        Card.print_pretty_cards(self.front)
        print('Mid:')
        Card.print_pretty_cards(self.mid)
        print('Back:')
        Card.print_pretty_cards(self.back)

    def get_free_streets(self):
        """Return a binary list of available streets, FMB."""
        available = [
            1 if len(self.front) < 3 else 0,
            1 if len(self.mid) < 5 else 0,
            1 if len(self.back) < 5 else 0
        ]

        return available

    def get_free_street_indices(self):
        available = []
        if len(self.front) < 3:
            available.append(0)
        if len(self.mid) < 5:
            available.append(1)
        if len(self.back) < 5:
            available.append(2)
        return available

    def place_card_by_id(self, card, street_id):
        if street_id == 0:
            self.front.append(card)
        if street_id == 1:
            self.mid.append(card)
        if street_id == 2:
            self.back.append(card)

    def is_complete(self):
        if len(self.back) == 5 and \
               len(self.mid) == 5 and \
               len(self.front) == 3:
            return True
        return False

    def is_foul(self, evaluator):
        if not self.is_complete():
            return False

        _ = []
        if evaluator.evaluate(self.front, _) >= \
               evaluator.evaluate(self.mid, _) >= \
               evaluator.evaluate(self.back, _):
            return False
        return True

    def get_combined(self):
        combined = (to_int53(self.front, 3), to_int53(self.mid, 5), to_int53(self.back, 5))
        assert len(combined[0]) == 3
        assert len(combined[1]) == 5
        assert len(combined[2]) == 5
        return combined


class OFCPokerBoard():

    def __init__(self, seed=-1, draw_eps=0.01):
        if seed > 0:
            random.seed(seed)

        self.deck = Deck()
        self.evaluator = OFCEvaluator()

        # Board for player (-1, 1)
        self.boards = {
            -1: OFCBoard(),
            1: OFCBoard()
        }
        self.current_cards = {
            -1: self.deck.draw(),
            1: self.deck.draw()
        }
        self.draw_eps = draw_eps

    def has_legal_moves(self, player):
        assert player in self.boards
        board = self.boards[player]
        return not board.is_complete() and not board.is_foul(self.evaluator)

    def get_legal_moves(self, player):
        assert player in self.boards
        board = self.boards[player]
        return board.get_free_streets() 

    def execute_move(self, street, player):
        assert player in self.boards
        assert player in self.current_cards
        board = self.boards[player]
        cur_card = self.current_cards[player]
        board.place_card_by_id(cur_card, street)

        # Replace card
        self.current_cards[player] = self.deck.draw()

    def to_numpy(self, player):
        """
        Convert to numpy array to feed into model
        """
        cur_card = card_to_int(self.current_cards[player])
        my_board = self.boards[player].get_combined()
        opp_board = self.boards[-player].get_combined()
        front = [np.sum(np.eye(53)[my_board[0]], axis=0), np.sum(np.eye(53)[opp_board[0]], axis=0)]
        mid = [np.sum(np.eye(53)[my_board[1]], axis=0), np.sum(np.eye(53)[opp_board[1]], axis=0)]
        back = [np.sum(np.eye(53)[my_board[2]], axis=0), np.sum(np.eye(53)[opp_board[2]], axis=0)]
        card = np.eye(53)[cur_card]
        return np.array(front), np.array(mid), np.array(back), card

    def to_torch(self, player):
        """
        Convert board into PyTorch tensor
        """
        f, m, b, c = self.to_numpy(player)
        return torch.Tensor(f), torch.Tensor(m), torch.Tensor(b), torch.Tensor(c)

    def to_tf(self, player):
        pass

    def __repr__(self):
        str_format = "{p}: Front {front} \n Mid {mid} \n Back {back} \n"
        p_1_str = str_format.format(p=1,
                                    front=[card_to_int(i) for i in self.boards[1].front],
                                    mid=[card_to_int(i) for i in self.boards[1].mid],
                                    back=[card_to_int(i) for i in self.boards[1].back])
        p_2_str = str_format.format(p=-1,
                                    front=[card_to_int(i) for i in self.boards[-1].front],
                                    mid=[card_to_int(i) for i in self.boards[-1].mid],
                                    back=[card_to_int(i) for i in self.boards[-1].back])
        return p_1_str + p_2_str

    def game_ended(self):
        if self.boards[1].is_foul(self.evaluator) or self.boards[-1].is_foul(self.evaluator):
            return True
        else:
            return self.boards[1].is_complete() and self.boards[-1].is_complete()

    def get_score(self, player):
        # +1 for each row for player
        # -1 for each row for -player
        assert self.game_ended()
        my_board = self.boards[player]
        opp_board = self.boards[-player]

        # fouls automatically go to the opponent
        if my_board.is_foul(self.evaluator):
            return -1
        elif opp_board.is_foul(self.evaluator):
            return 1
        else:
            score = 0

            # eval front

            my_front = self.evaluator.evaluate(my_board.front, [])
            opp_front = self.evaluator.evaluate(opp_board.front, [])
            # eval mid

            my_mid = self.evaluator.evaluate(my_board.mid, [])
            opp_mid = self.evaluator.evaluate(opp_board.mid, [])
            # eval back

            my_back = self.evaluator.evaluate(my_board.back, [])
            opp_back = self.evaluator.evaluate(opp_board.back, [])

            def eval_score(p1, p2):
                if p1 > p2:
                    return 1
                elif p2 > p1:
                    return -1
                else:
                    return 0

            score += eval_score(my_front, opp_front)
            score += eval_score(my_mid, opp_mid)
            score += eval_score(my_back, opp_back)

            if score == 0:
                score += self.draw_eps

            return score
