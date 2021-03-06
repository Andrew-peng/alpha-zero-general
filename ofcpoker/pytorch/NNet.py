import os
import shutil
import time
import random
import numpy as np
import math
import sys
sys.path.append('../../')
from utils import *
from pytorch_classification.utils import Bar, AverageMeter
from NeuralNet import NeuralNet

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from .OFCPokerNNet import OFCPokerNet as ofcpnet


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class NNetWrapper(NeuralNet):
    def __init__(self, game, embedding=256, lr=0.001, epochs=10, batch_size=64, drop_prob=0.3):
        self.input_dim, _ = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.nnet = ofcpnet(embedding, self.action_size, drop_prob)
        self.nnet = self.nnet.to(device)
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr

    def train(self, examples):
        """
        examples: list of examples, each example is of form (board, pi, v)
        """
        optimizer = optim.Adam(self.nnet.parameters(), lr=self.lr)

        for epoch in range(self.epochs):
            print('EPOCH ::: ' + str(epoch+1))
            self.nnet.train()
            data_time = AverageMeter()
            batch_time = AverageMeter()
            pi_losses = AverageMeter()
            v_losses = AverageMeter()
            end = time.time()

            bar = Bar('Training Net', max=int(len(examples)/self.batch_size))
            batch_idx = 0

            while batch_idx < int(len(examples)/self.batch_size):
                sample_ids = np.random.randint(len(examples), size=self.batch_size)
                boards, pis, vs = list(zip(*[examples[i] for i in sample_ids]))
                fronts, mids, backs, cards = [], [], [], []
                for b in boards:
                    fronts.append(b[0])
                    mids.append(b[1])
                    backs.append(b[2])
                    cards.append(b[3])
                fronts = [b[0] for b in boards]
                mids = [b[1] for b in boards]
                backs = [b[2] for b in boards]
                cards = [b[3] for b in boards]

                fronts = torch.FloatTensor(fronts).to(device)
                mids = torch.FloatTensor(mids).to(device)
                backs = torch.FloatTensor(backs).to(device)
                cards = torch.FloatTensor(cards).to(device)
                target_pis = torch.FloatTensor(np.array(pis)).to(device)
                target_vs = torch.FloatTensor(np.array(vs).astype(np.float64)).to(device)

                
                # measure data loading time
                data_time.update(time.time() - end)

                # compute output
                out_pi, out_v = self.nnet(fronts, mids, backs, cards)
                l_pi = self.loss_pi(target_pis, out_pi)
                l_v = self.loss_v(target_vs, out_v)
                total_loss = l_pi + l_v

                # record loss
                pi_losses.update(l_pi.item(), fronts.size(0))
                v_losses.update(l_v.item(), fronts.size(0))

                # compute gradient and do SGD step
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()
                batch_idx += 1

                # plot progress
                bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss_pi: {lpi:.4f} | Loss_v: {lv:.3f}'.format(
                            batch=batch_idx,
                            size=int(len(examples)/self.batch_size),
                            data=data_time.avg,
                            bt=batch_time.avg,
                            total=bar.elapsed_td,
                            eta=bar.eta_td,
                            lpi=pi_losses.avg,
                            lv=v_losses.avg,
                            )
                bar.next()
            bar.finish()


    def predict(self, fronts, mids, backs, card):
        """
        board: np array with board
        """
        # timing
        start = time.time()

        # preparing input
        fronts = torch.unsqueeze(fronts, 0).to(device)
        mids = torch.unsqueeze(mids, 0).to(device)
        backs = torch.unsqueeze(backs, 0).to(device)
        card = torch.unsqueeze(card, 0).to(device)
        self.nnet.eval()
        with torch.no_grad():
            pi, v = self.nnet(fronts, mids, backs, card)

        #print('PREDICTION TIME TAKEN : {0:03f}'.format(time.time()-start))
        return torch.exp(pi).data.cpu().numpy()[0], v.data.cpu().numpy()[0]

    def loss_pi(self, targets, outputs):
        return -torch.sum(targets*outputs)/targets.size()[0]

    def loss_v(self, targets, outputs):
        return torch.sum((targets-outputs.view(-1))**2)/targets.size()[0]

    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print("Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.mkdir(folder)
        else:
            print("Checkpoint Directory exists! ")
        torch.save({
            'state_dict' : self.nnet.state_dict(),
        }, filepath)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        # https://github.com/pytorch/examples/blob/master/imagenet/main.py#L98
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise("No model in path {}".format(filepath))
        map_location = device
        checkpoint = torch.load(filepath, map_location=map_location)
        self.nnet.load_state_dict(checkpoint['state_dict'])
