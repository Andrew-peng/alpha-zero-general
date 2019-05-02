import argparse

from Coach import Coach
from ofcpoker.OFCPokerGame import OFCPokerGame as Game
from ofcpoker.pytorch.NNet import NNetWrapper as nn
from utils import *


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-iter', type=int, default=1000,
                        help='Number of iterations')
    parser.add_argument('--num-episodes', type=int, default=100,
                        help='Number of episodes per iteration')
    parser.add_argument('--temp-threshold', type=int, default=15,)
    parser.add_argument('--update-threshold', type=float, default=0.6,)
    parser.add_argument('--max-queue-len', type=int, default=200000)
    parser.add_argument('--num-sims', type=int, default=40)
    parser.add_argument('--arena-compare', type=int, default=40)
    parser.add_argument('--cpuct', type=int, default=1)
    parser.add_argument('--checkpoint', type=str, default='./tmp/')
    parser.add_argument('--load-model', action='store_true')
    parser.add_argument('--history-iter', type=int, default=20)
    parser.add_argument('--load-folder-file', type=str, nargs='+', default=['./dev/models/8x100x50','best.pth.tar'])
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    args = parser.parse_args()

    g = Game()
    nnet = nn(g, epochs=args.epochs, batch_size=args.batch_size, lr=args.learning_rate)

    if args.load_model:
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])

    c = Coach(g, nnet, args)
    if args.load_model:
        print("Load trainExamples from file")
        c.loadTrainExamples()
    c.learn()
