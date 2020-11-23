# Most of this code is from https://github.com/ultmaster/neuralpredictor.pytorch 
# which was authored by Yuge Zhang, 2020
import logging
import random
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from naszilla.gcn.model import NeuralPredictor
from naszilla.gcn.utils import AverageMeterGroup, reset_seed, denormalize

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('device:', device)


def accuracy_mse(prediction, target, scale=100.):
    prediction = denormalize(prediction.detach()) * scale
    target = denormalize(target) * scale
    return F.mse_loss(prediction, target)


def fit(net,
        xtrain, 
        gcn_hidden=144,
        seed=0,
        batch_size=7,
        epochs=300,
        lr=1e-4,
        wd=3e-4):

    reset_seed(seed)

    data_loader = DataLoader(xtrain, batch_size=batch_size, shuffle=True, drop_last=True)

    net.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    net.train()

    for epoch in range(epochs):
        meters = AverageMeterGroup()
        lr = optimizer.param_groups[0]["lr"]
        for step, batch in enumerate(data_loader):
            target = batch["val_acc"]
            prediction = net(batch)
            loss = criterion(prediction, target)
            loss.backward()
            optimizer.step()
            mse = accuracy_mse(prediction, target)
            meters.update({"loss": loss.item(), "mse": mse.item()}, n=target.size(0))

        lr_scheduler.step()


def predict(net, xtest, eval_batch_size=1000):

    test_data_loader = DataLoader(xtest, batch_size=eval_batch_size)

    net.eval()
    meters = AverageMeterGroup()
    prediction_, target_ = [], []
    with torch.no_grad():
        for step, batch in enumerate(test_data_loader):
            prediction = net(batch)
            prediction_.append(prediction.cpu().numpy())

    prediction_ = np.concatenate(prediction_)

    def normalized_acc_to_loss(acc):
        return 100*(1 - denormalize(acc))

    losses = [normalized_acc_to_loss(acc) for acc in prediction_]
    return losses



