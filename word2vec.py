import numpy as np
from sklearn.linear_model import LinearRegression
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torch.utils.data.dataset import random_split
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
#stepbystep framework used (but simpler version used, from https://github.com/dvgodoy/PyTorchStepByStep/)
from neuralNetwork import StepByStep
from simplifyDataset import loadSMS
from collections import Counter
from nltk.tokenize import sent_tokenize, word_tokenize
import gensim
from gensim.models import word2vec
#This code used https://datajenius.com/2022/03/13/a-deep-dive-into-nlp-tokenization-encoding-word-embeddings-sentence-embeddings-word2vec-bert/ as a resource, but different application