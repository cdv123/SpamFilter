import numpy as np
from sklearn.linear_model import LinearRegression
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torch.utils.data.dataset import random_split
from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt

#Model
model = nn.Sequential(
    nn.Linear(1, 1)
)
optim.Adam(model.parameters(), lr = 0.0)

our_model = 
#One-hot encoding