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



#This code used https://datajenius.com/2022/03/13/a-deep-dive-into-nlp-tokenization-encoding-word-embeddings-sentence-embeddings-word2vec-bert/ as a resource, but different application

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

def convertSpamToBinary(spamList):
    for i in range(len(spamList)):
        if spamList[i] == 'ham':
            spamList[i] = 0
        else:
            spamList[i] = 1
    return spamList
trainingData,spamListTrain = loadSMS('SMSSpamcollection.csv')
valData,spamListVal = loadSMS('SMSVal.csv')
testData,spamListTest = loadSMS('SMSTest.csv')
spamListTrain = np.array(convertSpamToBinary(spamListTrain))
spamListTrain = spamListTrain.reshape(-1,1)
spamListVal = np.array(convertSpamToBinary(spamListVal))
spamListVal = spamListVal.reshape(-1,1)
spamListTest = convertSpamToBinary(spamListTest)
# spamListTest2 = np.array(spamListTest)
# spamListTest.reshape(-1,1)
mostCommonWords = getMostCommonWords(trainingData,300)
oneHotTrain = oneHotEncode(trainingData,mostCommonWords).values()
oneHotTrain = np.array([np.array(i) for i in oneHotTrain])
oneHotVal = oneHotEncode(valData,mostCommonWords).values()
oneHotVal = np.array([np.array(i) for i in oneHotVal])
oneHotTest = oneHotEncode(testData,mostCommonWords).values()
oneHotTest = np.array([np.array(i) for i in oneHotTest])


# Model
model = nn.Sequential(
    nn.Linear(300, 1)
)
loss_fn = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr = 0.01)
my_model = StepByStep(model, loss_fn, optimizer)
oneHotTrain = torch.as_tensor(oneHotTrain).float()
spamListTrain = torch.as_tensor(spamListTrain).float()
oneHotVal = torch.as_tensor(oneHotVal).float()
spamListVal = torch.as_tensor(spamListVal).float()
oneHotTest = torch.as_tensor(oneHotTest).float()
# spamListTest = torch.as_tensor(oneHotTest).float()
train_dataset = TensorDataset(oneHotTrain,spamListTrain)
val_dataset = TensorDataset(oneHotVal,spamListVal)
# test_dataset = TensorDataset(oneHotTest,spamListTest)
train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=32)
print(train_loader)
my_model.set_loaders(train_loader, val_loader)
my_model.train(100)
fig = my_model.plot_losses()
plt.show()
my_model.model.eval()
out = my_model.model(oneHotTest)
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
    if predictions[i] == spamListTest[i]:
        correct+=1

print('accuarcy', correct/len(predictions))