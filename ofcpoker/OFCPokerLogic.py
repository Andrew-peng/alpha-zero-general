import random

from deuces import Card, Deck, Evaluator


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
        Card.print_pretty_cards(self.front.cards)
        print('Mid:')
        Card.print_pretty_cards(self.mid.cards)
        print('Back:')
        Card.print_pretty_cards(self.back.cards)

    def get_free_streets(self):
        """Return a binary list of available streets, FMB."""
        available = [
            1 if self.front.length() < 3 else 0,
            1 if self.mid.length() < 5 else 0,
            1 if self.back.length() < 5 else 0
        ]

        return available

    def get_free_street_indices(self):
        available = []
        if self.front.length() < 3:
            available.append(0)
        if self.mid.length() < 5:
            available.append(1)
        if self.back.length() < 5:
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
            return True

        _ = []
        if evaluator.evaluate(self.front, _) >= \
               evaluator.evaluate(self.mid, _) >= \
               evaluator.evaluate(self.back, _):
            return False

        return True


class OFCPokerBoard():

    def __init__(self, seed=-1):
        if seed > 0:
            random.seed(seed)

        self.deck = Deck()
        self.evaluator = Evaluator()

        # Board for player (-1, 1)
        self.boards = {
            -1: OFCBoard(),
            1: OFCBoard()
        }
        self.current_cards = {
            -1: self.deck.draw(),
            1: self.deck.draw()
        }

    def has_legal_moves(self, player):
        assert player in self.boards
        board = self.boards[player]
        return not board.is_complete() and not board.is_foul()

    def get_legal_moves(self, player):
        assert player in self.boards
        board = self.boards[player]
        return board.get_free_street_indices() 

    def execute_move(self, street, player):
        assert player in self.boards
        assert player in self.current_cards
        board = self.boards[player]
        cur_card = self.current_cards[player]
        board.place_card_by_id(cur_card, street)

        # Replace card
        self.current_cards[player] = self.deck.draw()

    def to_numpy(self):
        """
        Convert to numpy array to feed into model
        """
        pass