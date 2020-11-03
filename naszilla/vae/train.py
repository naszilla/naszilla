# Most of this code is from https://github.com/muhanzhang/D-VAE
# which was authored by Muhan Zhang, 2019

from __future__ import print_function
import os
import sys
import math
import pickle
import gc
import pdb
import argparse
import random
from tqdm import tqdm
from shutil import copy
import torch
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import scipy.io
from scipy.linalg import qr 
import igraph
from random import shuffle
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from vae.util import *
from vae.models import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('device:', device)


MAX_N = 9
NUM_VERTEX_TYPE = 6 + 2 # 3 reg + 2 of my start/end + 2 others
START_TYPE = 0
END_TYPE = 1

def run_vae(xtrain, 
            xtest, 
            seed, 
            epochs=300,
            hs=501,
            nz=56,
            batch_size=32):

    np.random.seed(seed)
    random.seed(seed)

    train_data = load_nasbench(xtrain)
    test_data = load_nasbench(xtest)

    # VAE encoder

    model = eval('DVAE')(
            MAX_N, 
            NUM_VERTEX_TYPE, 
            START_TYPE, 
            END_TYPE, 
            hs=hs, 
            nz=nz, 
            bidirectional=False
            )

    # predictor
    predictor = nn.Sequential(
            nn.Linear(nz, hs), 
            nn.Tanh(), 
            nn.Linear(hs, 1)
            )
    model.predictor = predictor
    model.mseloss = nn.MSELoss(reduction='sum')

    model.to(device)

    # optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=10, verbose=True)

    '''Training begins here'''
    min_loss = math.inf
    min_loss_epoch = None

    for epoch in range(1, epochs + 1):

        train_loss, recon_loss, kld_loss, pred_loss = train(model,
                                                            train_data, 
                                                            optimizer,
                                                            epoch, 
                                                            batch_size)

        scheduler.step(train_loss)

    '''Testing begins here'''

    infer_batch_size = 32
    # test recon accuracy
    model.eval()
    encode_times = 10
    decode_times = 10
    Nll = 0
    pred_loss = 0
    n_perfect = 0

    print('Testing begins')
    g_batch = []
    y_batch = []
    y_pred = []
    for i, (g, y) in enumerate(test_data):

        g_batch.append(g)
        y_batch.append(y)
        if len(g_batch) == infer_batch_size or i == len(test_data) - 1:
            g = model._collate_fn(g_batch)
            mu, logvar = model.encode(g)
            
            _, nll, _ = model.loss(mu, logvar, g)
            
            # predictor:
            y_batch = torch.FloatTensor(y_batch).unsqueeze(1).to(device)

            # prediction happens here:
            batch_pred = model.predictor(mu)
            y_pred = [*y_pred, *batch_pred]
            
            batch_pred = []
            g_batch = []
            y_batch = []

    y_pred = [(1 - y.item()) * 100 for y in y_pred]

    model = None
    predictor = None
    gc.collect()

    return y_pred

def train(model,
          train_data, 
          optimizer,
          epoch, 
          batch_size):

    model.train()
    train_loss = 0
    recon_loss = 0
    kld_loss = 0
    pred_loss = 0
    shuffle(train_data)
    #pbar = tqdm(train_data)
    g_batch = []
    y_batch = []
    for i, (g, y) in enumerate(train_data):

        g_batch.append(g)
        y_batch.append(y)
        if len(g_batch) == batch_size or i == len(train_data) - 1:
            optimizer.zero_grad()
            g_batch = model._collate_fn(g_batch)

            mu, logvar = model.encode(g_batch)
            loss, recon, kld = model.loss(mu, logvar, g_batch)

            y_batch = torch.FloatTensor(y_batch).unsqueeze(1).to(device)
            y_pred = model.predictor(mu)
            pred = model.mseloss(y_pred, y_batch)
            loss += pred

            loss.backward()
            
            train_loss += float(loss)
            recon_loss += float(recon)
            kld_loss += float(kld)
            pred_loss += float(pred)
            optimizer.step()
            g_batch = []
            y_batch = []

    if epoch % 50 == 0:
        print('====> Epoch: {} Average loss: {:.4f}'.format(
              epoch, train_loss / len(train_data)))

    return train_loss, recon_loss, kld_loss, pred_loss
