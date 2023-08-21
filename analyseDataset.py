from word2vec import sentenceEmbedding
import numpy as np
from sklearn.linear_model import LinearRegression
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, TensorDataset, DataLoader
import matplotlib.pyplot as plt
#stepbystep framework used (but simpler version used, from https://github.com/dvgodoy/PyTorchStepByStep/)
from neuralNetwork import StepByStep
from simplifyDataset import loadSMS
from simplifyDataset import convertSpamToBinary
from oneHot import oneHotEncode
from oneHot import getMostCommonWords
from gensim import downloader
from gensim.models import word2vec
from word2vec import sentenceEmbedding
def trainNetwork(method_id,trainingData,trainingSpamList,vector_size=300,loss_fn = nn.BCEWithLogitsLoss()):
    model = nn.Sequential(
        nn.Linear(vector_size, 1)
    )   
    optimizer = optim.Adam(model.parameters(), lr = 0.01)
    my_model = StepByStep(model, loss_fn, optimizer)
    if method_id == 0:
        mostCommonWords = getMostCommonWords(trainingData,vector_size)
        oneHotTrain = oneHotEncode(trainingData,mostCommonWords).values()
        train_dataset = createDataset(oneHotTrain,trainingSpamList)
    elif method_id == 1:
        word2vec = downloader.load('word2vec-google-news-300')

    elif method_id == 2:
        pass
    train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
    my_model.set_loaders(train_loader)
    my_model.train(100)
    return my_model
def useNetwork(model,test):
    pass

def createDataset(data,spamList):
    spamList = np.array(convertSpamToBinary(spamList)).reshape(-1,1)
    data=np.array([np.array(i) for i in data])
    data = torch.as_tensor(data).float()
    spamList = torch.as_tensor(spamList).float()
    dataset = TensorDataset(data,spamList)
    return dataset
