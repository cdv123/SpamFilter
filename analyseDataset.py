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
from gensim.models import Word2Vec
from word2vec import sentenceEmbedding
from customEmbeddings import tokenize
from customEmbeddings import sentenceEmbedding2

def createDataset(data,spamList,valData,valSpamList):
    spamList = np.array(convertSpamToBinary(spamList)).reshape(-1,1)
    data=np.array([np.array(i) for i in data])
    data = torch.as_tensor(data).float()
    spamList = torch.as_tensor(spamList).float()
    dataset = TensorDataset(data,spamList)
    valSpamList = np.array(convertSpamToBinary(valSpamList)).reshape(-1,1)
    valData = np.array([np.array(i) for i in valData])
    valData = torch.as_tensor(valData).float()
    valSpamList = torch.as_tensor(valSpamList).float()
    return dataset,data,spamList,valData,valSpamList
def toTensor(data,spamList):
    spamList = np.array(convertSpamToBinary(spamList)).reshape(-1,1)
    data=np.array([np.array(i) for i in data])
    data = torch.as_tensor(data).float()
    spamList = torch.as_tensor(spamList).float()
    dataset = TensorDataset(data,spamList)
    return dataset,data,spamList
def trainNetwork(method_id,trainingData,trainingSpamList,valData,valSpamList,vector_size=500,loss_fn = nn.BCEWithLogitsLoss()):
    mostCommonWords =[]
    model = nn.Sequential(
        nn.Linear(vector_size, 1)
    )   
    optimizer = optim.Adam(model.parameters(), lr = 0.01)
    my_model = StepByStep(model, loss_fn, optimizer)
    if method_id == 0:
        mostCommonWords = getMostCommonWords(trainingData,vector_size)
        oneHotTrain = oneHotEncode(trainingData,mostCommonWords).values()
        oneHotVal = oneHotEncode(valData,mostCommonWords).values()
        train_dataset,oneHotTrain,trainingSpamList,oneHotVal,valSpamList = createDataset(oneHotTrain,trainingSpamList,oneHotVal,valSpamList)
        val_dataset = TensorDataset(oneHotVal,valSpamList)
    elif method_id == 1:
        word2vec = downloader.load('word2vec-google-news-300')
        mostCommonWords = word2vec
        trainSentences = sentenceEmbedding(trainingData,trainingSpamList,word2vec,vector_size)
        valSentences = sentenceEmbedding(valData,valSpamList,word2vec,vector_size)
        train_dataset,trainSentences,trainingSpamList,valSentences,valSpamList= createDataset(trainSentences,trainingSpamList,valSentences,valSpamList)
        val_dataset = TensorDataset(valSentences,valSpamList)
    elif method_id == 2:
        wordEmbedding = tokenize(trainingData)
        word2vec = Word2Vec(wordEmbedding,vector_size=vector_size,min_count=1,workers=4,window=5,sg=1)
        mostCommonWords = word2vec
        trainSentences = np.array(sentenceEmbedding2(trainingData,trainingSpamList,word2vec,vector_size))
        trainingSpamList = np.array(convertSpamToBinary(trainingSpamList)).reshape(-1,1)
        valSentences = np.array(sentenceEmbedding2(valData,valSpamList,word2vec,vector_size))
        valSpamList = np.array(convertSpamToBinary(valSpamList)).reshape(-1,1)
        trainSentences = torch.as_tensor(trainSentences).float()
        trainingSpamList = torch.as_tensor(trainingSpamList).float()
        valSentences = torch.as_tensor(valSentences).float()
        valSpamList = torch.as_tensor(valSpamList).float()
        train_dataset = TensorDataset(trainSentences,trainingSpamList)
        val_dataset = TensorDataset(valSentences,valSpamList)
    train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=32)
    my_model.set_loaders(train_loader,val_loader)
    my_model.train(100)
    return my_model,mostCommonWords
def useNetwork(my_model,testData,testSpamList,method_id,vector_size,mostCommonWords):
    my_model.model.eval()
    if method_id == 0:
        oneHotTest = oneHotEncode(testData,mostCommonWords).values()
        test_dataset,oneHotTest,testSpamList = toTensor(oneHotTest,testSpamList)
        out = my_model.model(oneHotTest)
    elif method_id == 1:
        testSentences = sentenceEmbedding(testData,testSpamList,mostCommonWords,vector_size)
        test_dataset,testSentences,testSpamList = toTensor(testSentences,testSpamList)
        out = my_model.model(testSentences)
    elif method_id == 2:
        testSentences = sentenceEmbedding2(testData,testSpamList,mostCommonWords,vector_size)
        testSpamList = convertSpamToBinary(testSpamList)
        testSentences = np.array([np.array(i) for i in testSentences])
        testSentences = torch.as_tensor(testSentences).float()
        out = my_model.model(testSentences)
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
    print("accuarcy:", correct/len(predictions))
    return predictions
trainingData,trainingSpamList = loadSMS("SMSSpamCollection.csv")
valData,valSpamList = loadSMS("SMSVal.csv")
testData,testSpamList = loadSMS("SMSTest.csv")
my_model,word2vec = trainNetwork(2,trainingData,trainingSpamList,valData,valSpamList)
useNetwork(my_model,testData,testSpamList,2,500,word2vec)