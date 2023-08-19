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
from gensim import corpora, downloader
from gensim.parsing.preprocessing import *
from gensim.utils import simple_preprocess
from gensim.models import Word2Vec
from gensim import downloader

#This code used https://datajenius.com/2022/03/13/a-deep-dive-into-nlp-tokenization-encoding-word-embeddings-sentence-embeddings-word2vec-bert/ as a resource, but different application

#Download vocab

word2vec = downloader.load('word2vec-google-news-300')

trainingData, trainingSpamList = loadSMS('SMSSpamCollection.csv')
valData, valSpamList = loadSMS('SMSVal.csv')
testData, testSpamList = loadSMS('SMSTest.csv')

#average out all word embeddings into sentence embedding
def sentenceEmbedding(data,spamList):
    sentenceEmbeddings = []
    dataLength = len(data)
    i = 0
    while i < dataLength:
        sentence = np.zeros(300)
        row = data[i].split(" ")
        length = 0
        for j in row:
            try:
                sentence = np.add(sentence,np.array(word2vec[j]))
                length+=1
            except:
                pass
        
        if length ==0:
            data.pop(i)
            spamList.pop(i)
            dataLength-=1
            i-=1
        else:
            for dim in sentence:
                dim = dim/length
            sentenceEmbeddings.append(sentence)
        i+=1
    return sentenceEmbeddings

def convertSpamToBinary(spamList):
    for i in range(len(spamList)):
        if spamList[i] == 'ham':
            spamList[i] = 0
        else:
            spamList[i] = 1
    return spamList

trainSentences = sentenceEmbedding(trainingData,trainingSpamList)
trainingSpamList = np.array(convertSpamToBinary(trainingSpamList))
trainingSpamList = trainingSpamList.reshape(-1,1)
trainSentences = np.array(trainSentences)
valSentences = sentenceEmbedding(valData,valSpamList)
valSpamList = np.array(convertSpamToBinary(valSpamList))
valSpamList = valSpamList.reshape(-1,1)
valSentences = np.array(valSentences)
testSentences = sentenceEmbedding(testData,testSpamList)
testSpamList = convertSpamToBinary(testSpamList)
testSentences = np.array([np.array(i) for i in testSentences])
# Model
model = nn.Sequential(
    nn.Linear(300, 1)
)
torch.manual_seed(42)
loss_fn = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr = 0.01)
my_model = StepByStep(model, loss_fn, optimizer)
trainSentences = torch.as_tensor(trainSentences).float()
trainingSpamList = torch.as_tensor(trainingSpamList).float()
valSentences = torch.as_tensor(valSentences).float()
valSpamList = torch.as_tensor(valSpamList).float()
testSentences = torch.as_tensor(testSentences).float()
print(trainSentences.shape)
print(trainingSpamList.shape)
print(valSentences.shape)
print(valSpamList.shape)
train_dataset = TensorDataset(trainSentences,trainingSpamList)
val_dataset = TensorDataset(valSentences,valSpamList)
# test_dataset = TensorDataset(oneHotTest,spamListTest)
train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=32)
print(model)
my_model.set_loaders(train_loader, val_loader)
my_model.train(100)
fig = my_model.plot_losses()
plt.show()
my_model.model.eval()
out = my_model.model(testSentences)
print(out)
prob_sigmoid = torch.sigmoid(out).squeeze().tolist()
predictions = []
for i in prob_sigmoid:
    if i > 0.5:
        predictions.append(1)
    else:
        predictions.append(0)
correct = 0
for i in range(len(predictions)):
    if predictions[i] == testSpamList[i]:
        correct+=1
print('accuarcy', correct/len(predictions))

