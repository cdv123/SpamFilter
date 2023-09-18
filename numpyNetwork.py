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
def neuralNetwork2(mini_batches,val_mini_batches,lr,epochs):
    np.random.seed(0)
    random.seed(0)
    vectorSize = len(mini_batches[0][0][0])
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
        for x in range(len(mini_batches)):
            y = x
            while y >= len(val_mini_batches):
                y-= len(val_mini_batches)
            gradw = np.zeros(vectorSize)
            gradb = 0
            for i in range(len(mini_batches[x])):
                j = i
                while j >= len(val_mini_batches[y]):
                    j-=len(val_mini_batches[y])
                # Compute prediction
                a = getOutput(mini_batches[x][i][0],w,b)
                a = sigmoid(a)
                a_val = getOutput(val_mini_batches[y][j][0],w,b)
                a_val = sigmoid(a_val)
                # activation function - normalise value
                error[epoch]+= BCE(mini_batches[x][i][1],a) 
                error_val[epoch]+= BCE(val_mini_batches[y][j][1],a_val)
                # compute gradient loss (actual loss not needed)
                new_gradw,new_gradb = computeGradient(mini_batches[x][i][0],mini_batches[x][i][1],a)
                gradb+=new_gradb
                gradw+=new_gradw
                # update weights using gradient loss
            gradb = gradb/len(mini_batches[x])
            gradw = gradw/len(mini_batches[x])
            w,b = updateWeights(w,b,gradw,gradb,lr)
        error[epoch] = error[epoch]/len(trainingData)
        error_val[epoch] = error_val[epoch]/len(trainingData)
    return w,b,error,error_val
def sigmoid(x):
    return (1/(1+(np.exp(-x))))

def getBatchOutput(mini_batch,w,b):
    a = [sigmoid(getOutput(x[0],w,b)) for x in mini_batch]
    return a

def sigmoidDerivative(yhat):
    return (yhat)*(1-yhat)

def BCE(y,logit):
    # binary cross entroppy loss function
    loss = np.multiply(-y,np.log(logit))-np.multiply((1-y),np.log(1-logit))
    return loss

def BCEbatch(mini_batch,a):
    err_list = [BCE(mini_batch[x][1],a[x]) for x in range(len(mini_batch))]
    avg_err = np.sum(err_list)*1/len(mini_batch)
    return avg_err
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

def computeBatchGradient(mini_batch,a):
    gradArr = np.array([computeGradient(mini_batch[x][0],mini_batch[x][1],a[x]) for x in range(len(mini_batch))])
    gradx = np.sum(gradArr[:,0])/len(mini_batch)
    gradb = np.sum(gradArr[:,1])/len(mini_batch)
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

def make_mini_batches(trainingData,spamData,batch_size):
    mini_batches = []
    i = 0
    while i < len(trainingData):
        mini_batches.append([])
        for j in range(batch_size):
            if i >= len(trainingData):
                break
            mini_batches[-1].append((trainingData[i],spamData[i]))
            i+=1
    return mini_batches

def twoLayerNetwork(trainingData,spamData,lr,hiddenNodesNum,epochs,val_data,val_spam):
    vectorSize = len(trainingData[0])
    np.random.seed(0)
    limit= math.sqrt(6/(vectorSize+hiddenNodesNum))
    # limit = math.sqrt(2/vectorSize)
    # weight matrix to go from input layer to hidden layer
    hiddenW = np.random.normal(0,limit,size=(vectorSize,hiddenNodesNum))
    # hiddenW = np.random.uniform(-limit,limit,size=(vectorSize,hiddenNodesNum))
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
            # contribution of hidden layer to error
            errorHidden = np.multiply(hiddenW*outputError,derReLu(hiddenRes))
            hiddenRes = ReLu(hiddenRes)
            # error gradient of outputW and outputB
            gradOutputW, gradOutputB = computeGradient(hiddenRes,spamData[x],finalRes)
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
    arr=[1 if i >= 0  else 0 for i in x]
    return arr

def forward_pass(x,hiddenW,hiddenBias,outputW,outputBias):
    hiddenOut = np.matmul(x,hiddenW)
    hiddenOut+= hiddenBias
    # hiddenOut = sigmoid(hiddenOut)
    # hiddenOut = ReLu(hiddenOut)
    finalOut = np.dot(hiddenOut,outputW)+outputBias
    return hiddenOut,finalOut

def findResult(x,hiddenW,hiddenBias,outputW,outputBias):
    hiddenOut = np.matmul(x,hiddenW)+hiddenBias
    # hiddenOut = sigmoid(hiddenOut)
    hiddenOut = ReLu(hiddenOut)
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
vector_size = 100
embeddingDict = useEmbedding2()
trainSentences = sentenceEmbedding(trainingData,spamData,embeddingDict,100)
valSentences = sentenceEmbedding(valData,spamVal,embeddingDict,100)
mini_batches = make_mini_batches(trainSentences,spamData,10)
val_batches = make_mini_batches(valSentences,spamVal,10)
# # mostCommonWords = getMostCommonWords(trainingData,vector_size)
# # oneHotTrain = list(oneHotEncode(trainingData,mostCommonWords).values())
# # oneHotVal = list(oneHotEncode(valData,mostCommonWords).values())
# # hiddenW,hiddenBias,weights,bias,error,error_val = twoLayerNetwork(oneHotTrain,spamData,0.00025,9,20,oneHotVal,spamVal)
# # hiddenW,hiddenBias,weights,bias,error,error_val = twoLayerNetwork(oneHotTrain,spamData,0.004,9,10,oneHotVal,spamVal)
# hiddenW,hiddenBias,weights,bias,error,error_val = twoLayerNetwork(trainSentences,spamData,0.004,9,10,valSentences,spamVal)
# weights,bias,error,error_val = neuralNetwork2(mini_batches,val_batches,0.008,20)
weights,bias,error,error_val = neuralNetwork(trainSentences,spamData,0.003,20,valSentences,spamVal)
# # weights,bias,error,error_val = neuralNetwork(oneHotTrain,spamData,0.03,10,oneHotVal,spamVal)
# # # print(bias)
plt.plot(range(20),error, label = "Training Loss")
plt.plot(range(20),error_val, label = "Validation Loss")
plt.legend()
plt.show()
testData, spamTest = loadSMS2('SMSTest.txt')
# # oneHotTest = list(oneHotEncode(testData,mostCommonWords).values())
spamTest = convertSpamToBinary(spamTest)
testSentences = sentenceEmbedding(testData,spamTest,embeddingDict,100)
# print(testTwoLayer(testSentences,spamTest,hiddenW,hiddenBias,weights,bias))
# print(testTwoLayer(oneHotTest,spamTest,hiddenW,hiddenBias,weights,bias))
print(testNetwork(testSentences,spamTest,weights,bias))
# print(testNetwork(oneHotTest,spamTest,weights,bias))


