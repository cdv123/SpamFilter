import numpy as np
import math
from simplifyDataset import loadSMS2,convertSpamToBinary,loadMessage
from oneHot import oneHotEncode,oneHotEncode2,getMostCommonWords
import random
from word2vec import useEmbedding2,sentenceEmbedding
import matplotlib.pyplot as plt
def neuralNetwork(trainingData,spamData,lr,epochs,val_data,val_spam):
    np.random.seed(0)
    random.seed(0)
    vectorSize = len(trainingData[0])
    #initialise weights
    w = np.random.rand(vectorSize)
    #initialise m and v for adam optimizer
    m = np.zeros(vectorSize)
    v = np.zeros(vectorSize)
    #initialise bias
    b = random.random()
    #train step - 
    error = [0]*epochs
    error_val = [0]*epochs
    for epoch in range(epochs):
        for x in range(len(trainingData)):
            y = x
            while y >= len(val_data):
                y-= len(val_data)
            # Compute prediction
            a = getOutput(trainingData[x],w,b)
            a_val = getOutput(val_data[y],w,b)
            # activation function - normalise value
            a = sigmoid(a)
            a_val = sigmoid(a_val)
            error[epoch]+= BCE(spamData[x],a) 
            error_val[epoch]+= BCE(val_spam[y],a_val)
            # compute gradient loss (actual loss not needed)
            gradw,gradb = computeGradient(trainingData[x],spamData[x],a)
            # update weights using gradient loss
            w,b = updateWeights(w,b,gradw,gradb,lr)
        error[epoch] = error[epoch]/len(trainingData)
        error_val[epoch] = error_val[epoch]/len(trainingData)
    return w,b,error,error_val

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

def twoLayerNetwork(trainingData,spamData,lr,hiddenNodesNum,epochs,val_data,val_spam):
    vectorSize = len(trainingData[0])
    np.random.seed(0)
    limit= math.sqrt(6/(vectorSize+hiddenNodesNum))
    # weight matrix to go from input layer to hidden layer
    hiddenW = np.random.uniform(-limit,limit,size=(vectorSize,hiddenNodesNum))
    # hiddenW = np.zeros((vectorSize,hiddenNodesNum))
    # bias vector to go from input layer to hidden layer
    hiddenBias = np.zeros(hiddenNodesNum)
    limit=math.sqrt(6/(1+hiddenNodesNum))
    # weight vector to go from hidden layer to output layer
    outputW = np.random.uniform(-limit,limit,size=hiddenNodesNum)
    error = [0]*epochs
    error_val = [0]*epochs
    # outputW =np.zeros(hiddenNodesNum)
    # bias scalar to go from hidden layer to output layer
    outputBias = 0
    for epoch in range(epochs):
        for x in range(len(trainingData)):
            y = x
            while y >= len(val_data):
                y-=len(val_data)
            # compute result of hidden layer and final output
            hiddenRes,finalRes = forward_pass(trainingData[x],hiddenW,hiddenBias,outputW,outputBias)
            valRes = forward_pass(val_data[y],hiddenW,hiddenBias,outputW,outputBias)[1]
            # activation function for final output to normalize
            finalRes = sigmoid(finalRes)
            valRes = sigmoid(valRes)
            error[epoch]+=BCE(spamData[x],finalRes)
            error_val[epoch]+=BCE(val_spam[y],valRes)
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
        error[epoch] = error[epoch]/len(trainingData)
        error_val[epoch] = error_val[epoch]/len(trainingData)
    return hiddenW,hiddenBias,outputW,outputBias,error,error_val

def computeHiddenGradient(errorHidden,vectorInput):
    gradHiddenW = np.dot(vectorInput,errorHidden)
    gradHiddenBias = np.sum(errorHidden,axis=0)
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

# def initializeWeights(vectorSize,hiddenNodesNum):
#     W = np.random.rand((x_dim,y_dim))*np.sqrt(1/(vectorSize+hiddenNodesNum))
#     outputW = np.random.uniform(-limit,limit,size=hiddenNodesNum)

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
trainingData,spamData = loadSMS2('SMSSpamCollection.txt')
spamData = convertSpamToBinary(spamData)
valData,spamVal = loadSMS2('SMSVal.txt')
spamVal = convertSpamToBinary(spamVal)
vector_size = 300
embeddingDict = useEmbedding2()
# trainSentences = sentenceEmbedding(trainingData,spamData,embeddingDict,100)
# valSentences = sentenceEmbedding(valData,spamVal,embeddingDict,100)
mostCommonWords = getMostCommonWords(trainingData,vector_size)
oneHotTrain = list(oneHotEncode(trainingData,mostCommonWords).values())
oneHotVal = list(oneHotEncode(valData,mostCommonWords).values())
hiddenW,hiddenBias,weights,bias,error,error_val = twoLayerNetwork(oneHotTrain,spamData,0.0005,9,20,oneHotVal,spamVal)
# hiddenW,hiddenBias,weights,bias,error,error_val = twoLayerNetwork(oneHotTrain,spamData,0.004,9,10,oneHotVal,spamVal)
# weights,bias,error,error_val = neuralNetwork(trainSentences,spamData,0.001,20,valSentences,spamVal)
# weights,bias,error,error_val = neuralNetwork(oneHotTrain,spamData,0.03,10,oneHotVal,spamVal)
# # print(bias)
plt.plot(range(20),error, label = "Training Loss")
plt.plot(range(20),error_val, label = "Validation Loss")
plt.legend()
plt.show()
testData, spamTest = loadSMS2('SMSTest.txt')
oneHotTest = list(oneHotEncode(testData,mostCommonWords).values())
spamTest = convertSpamToBinary(spamTest)
# testSentences = sentenceEmbedding(testData,spamTest,embeddingDict,100)
# print(testTwoLayer(testSentences,spamTest,hiddenW,hiddenBias,weights,bias))
print(testTwoLayer(oneHotTest,spamTest,hiddenW,hiddenBias,weights,bias))
# print(testNetwork(testSentences,spamTest,weights,bias))
# print(testNetwork(oneHotTest,spamTest,weights,bias))


