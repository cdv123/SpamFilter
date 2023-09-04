import numpy as np
import math
from simplifyDataset import loadSMS2,convertSpamToBinary,loadMessage
from oneHot import oneHotEncode,oneHotEncode2,getMostCommonWords
import random
def neuralNetwork(trainingData, spamData,lr):
    vectorSize = len(trainingData[0])
    #initialise weights
    w = np.random.rand(vectorSize)*lr
    #initialise m and v for adam optimizer
    m = np.zeros(vectorSize)
    v = np.zeros(vectorSize)
    #initialise bias
    b = random.random()

    # shuffle the lists with same order
    np.random.shuffle(trainingData)
    #train step - 
    for epoch in range(10):
        for x in range(len(trainingData)):
            # Compute prediction
            a = getOutput(trainingData[x],w,b)
            # activation function - normalise value
            a = sigmoid(a)
            error = BCE(spamData[x],a) 
            # compute gradient loss (actual loss not needed)
            gradw,gradb = computeGradient(trainingData[x],spamData[x],a)
            # update weights using gradient loss
            w,b = updateWeights(w,b,gradw,gradb,lr)
    return w,b

def sigmoid(x):
    return (1/(1+(np.exp(-x))))

def sigmoidDerivative(yhat):
    return (yhat)*(1-yhat)

def BCE(y,logit):
    # binary cross entroppy loss function
    loss = np.multiply(-y,np.log(logit))-np.multiply((1-y),np.log(1-logit))
    return loss

def getOutput(x,w,b):
    return (np.matmul(x,w)+b)

def BCEgrad(y,yhat):
    if y == 1:
        return -1/yhat
    return 1/(1-yhat)
def computeGradient(x,y,a):
    gradb = BCEgrad(y,a)*sigmoidDerivative(a)
    gradx = np.multiply(gradb,x)
    return gradx,gradb
def updateWeights(w,b,dw,db,lr):
    w = w-lr*dw
    b = b-lr*db
    return w,b

def adamOptim(w,lr,m,v,g):
    epsilon = 1*(10**-8)
    m = 0.99*m + (1-0.99)*g
    v = 0.999*v + (1-0.999)*(g^2)
    mhat = m/(1-0.99)
    vhat = m/(1-0.999)
    w = w-((lr)/math.sqrt(vhat)+epsilon)*mhat
    return m,v,w
def testNetwork(testData,spamTest,w,b):
    outputs = []
    correct = 0
    incorrect = 0
    for i in testData:
        outputs.append(getOutput(i,w,b))
    predictions = []
    for i in range(len(outputs)):
        if sigmoid(outputs[i])>0.5:
            if i == 11:
                print(testData[i],spamTest[i])
            predictions.append(1)
        else:
            predictions.append(0)
    for i in range(len(predictions)):
        if predictions[i] == spamTest[i]:
            correct+=1
        else:
            incorrect+=1
    accuarcy = correct/(correct+incorrect)
    return accuarcy

def twoLayerNetwork(trainingData,spamData,lr,hiddenNodesNum):
    vectorSize = len(trainingData[0])
    # weight matrix to go from input layer to hidden layer
    hiddenW = np.random.rand(vectorSize,hiddenNodesNum)
    # bias vector to go from input layer to hidden layer
    hiddenBias = np.random.rand(hiddenNodesNum)
    # weight vector to go from hidden layer to output layer
    outputW = np.random.rand(hiddenNodesNum)
    # bias scalar to go from hidden layer to output layer
    outputBias = random.random()
    np.random.shuffle(trainingData)
    for epoch in range(10):
        for x in range(len(trainingData)):
            # compute result of hidden layer and final output
            hiddenRes,finalRes = forward_pass(trainingData[x],hiddenW,hiddenBias,outputW,outputBias)
            # activation function for final output to normalize
            finalRes = sigmoid(finalRes)
            # binary cross entropy used to measure error, equation for ouput layer only equation dependent on cost function
            errorDer = BCEgrad(spamData[x],finalRes)
            outputError = np.multiply(errorDer,finalRes*(1-finalRes))
            # error gradient of outputW and outputB
            gradOutputW, gradOutputB = computeGradient(hiddenRes,spamData[x],finalRes)
            # contribution of hidden layer to error
            errorHidden = np.multiply(hiddenW*outputError,hiddenRes*(1-hiddenRes))
            # update parameters to go from hidden layer to output layer
            outputW,outputBias = updateWeights(outputW,outputBias,gradOutputW,gradOutputB,lr)
            # error gradient of hiddenW and hiddenB
            gradHiddenW,gradHiddenBias = computeHiddenGradient(errorHidden,trainingData[x])
            # update parameters to go grom input layer to output layer
            hiddenW, hiddenBias = updateWeights(hiddenW,hiddenBias,gradHiddenW,gradHiddenBias,lr)
            # print(error)
    return hiddenW,hiddenBias,outputW,outputBias

def computeHiddenGradient(errorHidden,vectorInput):
    gradHiddenW = np.dot(vectorInput,errorHidden)
    gradHiddenBias = np.sum(errorHidden)
    return gradHiddenW,gradHiddenBias

def ReLu(x):
    return np.maximum(0,x)         

def derReLu(x):
    arr=[1 if i > 0  else 0 for i in x]
    gradx = np.multiply(x,arr)
    return gradx

def forward_pass(x,hiddenW,hiddenBias,outputW,outputBias):
    hiddenOut = np.matmul(x,hiddenW)
    hiddenOut+= hiddenBias
    hiddenOut = sigmoid(hiddenOut)
    finalOut = np.dot(hiddenOut,outputW)+outputBias
    return hiddenOut,finalOut

def findResult(x,hiddenW,hiddenBias,outputW,outputBias):
    hiddenOut = np.matmul(x,hiddenW)+hiddenBias
    hiddenOut = sigmoid(hiddenOut)
    finalOut = np.dot(hiddenOut,outputW)+outputBias
    finalOut = sigmoid(finalOut)
    return finalOut

def testTwoLayer(testData,spamData,hiddenW,hiddenBias,outputW,outputBias):
    correctCount = 0
    for i in range(len(testData)):
        prediction = findResult(testData[i],hiddenW,hiddenBias,outputW,outputBias)
        if prediction > 0.5:
            prediction = 1
        else:
            prediction = 0
        if prediction == spamData[i]:
            correctCount+=1
    return correctCount/len(testData)
# newLines = ""
# with open('Dataset/SMSTest copy.txt') as file:
#     lines = file.readlines()
#     for i in lines:
#         if i[1] == "s":
#             newLines+= i
# print(len(newLines))
# with open('Dataset/SMSTest copy.txt','w') as file:
#     file.write(newLines)
trainingData,spamData = loadSMS2('SMSSpamCollection.txt')
spamData = convertSpamToBinary(spamData)
vector_size = 300
mostCommonWords = getMostCommonWords(trainingData,vector_size)
oneHotTrain = list(oneHotEncode(trainingData,mostCommonWords).values())
# weights,bias,hiddenW,hiddenBias = twoLayerNetwork(oneHotTrain,spamData,0.04,2)
weights,bias = neuralNetwork(oneHotTrain,spamData,0.001)
# print(bias)
testData, spamTest = loadSMS2('SMSTest.txt')
oneHotTest = list(oneHotEncode(testData,mostCommonWords).values())
spamTest = convertSpamToBinary(spamTest)
# print(testTwoLayer(oneHotTest,spamTest,weights,bias,hiddenW,hiddenBias))
print(testNetwork(oneHotTest,spamTest,weights,bias))
# message = "Had your contract mobile 11 Mnths? Latest Motorola, Nokia etc. all FREE! Double Mins & Text on Orange tariffs. TEXT YES for callback, no to remove from records"
# message = loadMessage(message)
# oneHotData = oneHotEncode2(message,mostCommonWords)
# print(sigmoid(getOutput(oneHotData,weights,bias)))
# trainingData, spamData = loadSMS('SMSSpamCollection.txt')
# spamData = convertSpamToBinary(spamData)
# vector_size = 300
# wordEmbedding = tokenize(trainingData)
# word2vec = Word2Vec(wordEmbedding,vector_size=vector_size,min_count=1,workers=4,window=5,sg=1)
# trainSentences = sentenceEmbedding2(trainingData,spamData,word2vec,vector_size)
# # mostCommonWords = getMostCommonWords(trainingData,vector_size)
# # oneHotTrain = list(oneHotEncode(trainingData,mostCommonWords).values())
# print(testData)
# testSentences = sentenceEmbedding2(testData,spamTest,word2vec,vector_size)
# # weights,bias = neuralNetwork(oneHotTrain,spamData,0.9)
# # print(weights,bias)
# weights,bias = neuralNetwork(trainSentences,spamData,0.9)
# print(testNetwork(testSentences,spamTest,weights,bias))

