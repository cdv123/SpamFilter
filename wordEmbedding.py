import numpy as np
from sklearn.linear_model import LinearRegression
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torch.utils.data.dataset import random_split
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from neuralNetwork import StepByStep
from simplifyDataset import loadSMS
from collections import Counter


# Model
model = nn.Sequential(
    nn.Linear(300, 1)
)
loss_fn = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr = 0.01)
our_model = StepByStep(model, loss_fn, optimizer)
our_model.set_loaders(train_loader, val_loader)
our_model.train(100)
#This code used https://datajenius.com/2022/03/13/a-deep-dive-into-nlp-tokenization-encoding-word-embeddings-sentence-embeddings-word2vec-bert/ as a resource, but different application

#Probality to value:
#If probability >= 50% => spam => positive value when using sigmoid function
#If probability < 50% => ham => positive value when using sigmoid function

# First implementation with One-Hot encoding

def getMostCommonWords(dataset,tokenNum):
    allWords = []
    for i in range(len(dataset)):
        row = dataset[i].split(" ")
        for word in row:
            if word != '': 
                allWords.append(word)
    word_count = Counter(allWords)
    word_count = word_count.most_common(tokenNum)
    mostCommonWords = []
    for i in range(0,len(word_count)):
        mostCommonWords.append(word_count[i][0])
    return mostCommonWords

def oneHotEncode(dataset,mostCommonWords):
    oneHot = {}
    for i in range(len(dataset)):
        oneHot[i] = [0]*len(mostCommonWords)
        row = dataset[i].split(" ")
        for j in range(len(mostCommonWords)):
            if mostCommonWords[j] in row:
                oneHot[i][j] = 1
    return oneHot

trainingData,spamList = loadSMS('SMSSpamcollection.csv')
mostCommonWords = getMostCommonWords(trainingData,300)
oneHot = oneHotEncode(trainingData,mostCommonWords)
for i in range(len(trainingData)):
    print(trainingData[i],oneHot[i])

# Second implementation with Vectors using word2Vec

# Third implementation with Vectors using custom implementation

# Fourth implementation with BERTft
