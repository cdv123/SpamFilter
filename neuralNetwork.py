import numpy as np
import datetime
import torch
import torch.optim as optim
import torch.nn as nn
import torch.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

#framework from StepByStep class by Daniel Voigt Godoy https://github.com/dvgodoy/PyTorchStepByStep

plt.style.use('fivethirtyeight')

class StepByStep(object):
    def __init__(self,model,loss_fn,optimizer):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        #attributes defined later on but not needed immediately
        self.train_loader = None
        self.val_loader = None
        self.writer = None
    #allows user to specify a different device
    def to(self,device):
        try:
            self.device = device
            self.model.to(self.device)
        except RuntimeError:
            self.device = ('cuda' if torch.cuda.is_available() else 'cpu')
            print(f"Couldn't send it to {device}, sending it to {self.device} instead.")
            self.model.to(self.device)
