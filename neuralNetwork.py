import numpy as np
import math
from word2vec import useEmbedding2,sentenceEmbedding
import matplotlib.pyplot as plt 
from oneHot import oneHotEncode,getMostCommonWords
from simplifyDataset import convertSpamToBinary,loadSMS2
class NeuralNetwork:    
    def __init__(self,layer_num):
        self.layer_num = layer_num
        self.weight_matrix = 0
        self.weight_vector = 0
        self.bias = 0
        self.bias_vector = 0
        self.error = []
        self.error_val = []
    def train_network(self,epochs,training_data,result_data,lr,val_data,val_spam,nodes_num=0):
        vector_size = len(training_data[0])
        self.error = [0]*epochs
        self.error_val = [0]*epochs
        if self.layer_num == 1:
            # train neural network with only input and output layer
            self.weight_matrix,self.bias_vector = self.weight_init(vector_size)
            for epoch in range(epochs):
                for x in range(len(training_data)):
                    y = x
                    while y >= len(val_data):
                        y-= len(val_data)
                    # Compute prediction
                    a = self.forward_pass(training_data[x])
                    a_val = self.forward_pass(val_data[y])
                    # Error measured using binary cross entropy
                    self.error[epoch] += BCE(result_data[x],a) 
                    self.error_val[epoch] += BCE(val_spam[y],a_val)
                    # compute gradient loss (actual loss not needed)
                    gradw,gradb = self.compute_gradient(training_data[x],result_data[x],a)
                    # update weights using gradient loss
                    self.weight_matrix,self.bias_vector = self.update_weights(self.weight_matrix,self.bias_vector,gradw,gradb,lr)
                self.error[epoch] = self.error[epoch]/len(training_data)
                self.error_val[epoch] = self.error_val[epoch]/len(training_data)
        else:
            # train neural network with input, one hidden layer and one output layer
            self.weight_matrix,self.weight_vector,self.bias,self.bias_vector = self.weight_init(vector_size,nodes_num)
            for epoch in range(epochs):
                for x in range(len(training_data)):
                    y = x
                    while y >= len(val_data):
                        y-= len(val_data)
                    #compute output of hidden layer and output layer
                    hidden_res,final_res = self.forward_pass(training_data[x])
                    val_res = self.forward_pass(val_data[y])[1]
                    self.error[epoch]+=BCE(result_data[x],final_res)
                    self.error_val[epoch]+=BCE(val_spam[y],val_res)
                    # binary cross entropy used to measure error, equation for ouput layer only equation dependent on cost function
                    error_der = BCEgrad(result_data[x],final_res)
                    output_error = np.multiply(error_der,final_res*(1-final_res))
                    # error gradient of outputW and outputB
                    grad_output_w, grad_output_b = self.compute_gradient(hidden_res,result_data[x],final_res)
                    # contribution of hidden layer to error
                    error_hidden = np.multiply(self.weight_matrix*output_error,hidden_res*(1-hidden_res))
                    # update parameters to go from hidden layer to output layer
                    self.weight_vector, self.bias = self.update_weights(self.weight_vector,self.bias,grad_output_w,grad_output_b,lr)
                    # error gradient of hiddenW and hiddenB
                    grad_hidden_w,grad_hidden_bias = self.compute_hidden_gradient(error_hidden,training_data[x])
                    # update parameters to go grom input layer to output layer
                    self.weight_matrix, self.bias_vector = self.update_weights(self.weight_matrix,self.bias_vector,grad_hidden_w,grad_hidden_bias,lr)
    def test_network(self,test_data,spam_data):
        correct_count = 0
        for i in range(len(test_data)):
            # compute prediction
            prediction = self.forward_pass(test_data[i])
            if self.layer_num == 2:
                # function returns tuple if layer num is 2, choose second 1 as this is the final result
                prediction = prediction[1]
            # if results is greater than 0.5, prediction is spam
            if prediction > 0.5:
                prediction = 1
            else:
                prediction = 0
            if prediction == spam_data[i]:
                correct_count+=1
        # return accuracy
        return correct_count/len(test_data)
    def forward_pass(self,x):
        hidden_res = np.matmul(x,self.weight_matrix)
        hidden_res+=self.bias_vector
        hidden_res = sigmoid(hidden_res)
        if self.layer_num == 1:
            return hidden_res
        final_res = np.dot(hidden_res,self.weight_vector)+self.bias
        final_res = sigmoid(final_res)
        return hidden_res,final_res
    def weight_init(self,vector_size,hidden_nodes_num=1):
        np.random.seed(0)
        limit = math.sqrt(6/(vector_size+hidden_nodes_num))
        bias = 0
        if self.layer_num == 1:
            weight = np.random.uniform(-limit,limit,size=(vector_size))
            return weight,bias
        weight = np.random.uniform(-limit,limit,size=(vector_size,hidden_nodes_num))
        limit = math.sqrt(6/(1+hidden_nodes_num))
        weight_vector = np.random.uniform(-limit,limit,size=(hidden_nodes_num))
        bias_vector = np.zeros(hidden_nodes_num)
        return weight,weight_vector,bias,bias_vector
    def update_weights(self,w,b,dw,db,lr):
        w = w-lr*dw
        b = b-lr*db
        return w,b
    def compute_gradient(self,x,y,a):
        gradb = BCEgrad(y,a)*sigmoid_derivative(a)
        gradx = np.multiply(gradb,x)
        return gradx,gradb
    def compute_hidden_gradient(self,error_hidden,vector_input):
        grad_hidden_w = np.dot(vector_input,error_hidden)
        grad_hidden_bias = np.sum(error_hidden,axis=0)
        return grad_hidden_w,grad_hidden_bias
    def plot(self,epochs):
        fig,ax = plt.subplots()
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Training Loss')
        ax.plot(range(epochs),self.error)
        ax.plot(range(epochs),self.error_val)
        return fig,ax
def sigmoid(x):
    return (1/(1+(np.exp(-x))))

def sigmoid_derivative(yhat):
    return (yhat)*(1-yhat)

def BCE(y,logit):
    # binary cross entroppy loss function
    loss = np.multiply(-y,np.log(logit))-np.multiply((1-y),np.log(1-logit))
    return loss

def BCEgrad(y,yhat):
    if y == 1:
        return -1/yhat
    return 1/(1-yhat)

# trainingData,spamData = loadSMS2('SMSSpamCollection.txt')
# valData,spamVal = loadSMS2('SMSVal.txt')
# spamData = convertSpamToBinary(spamData)
# vector_size = 100
# embeddingDict = useEmbedding2()
# trainSentences = sentenceEmbedding(trainingData,spamData,embeddingDict,100)
# mostCommonWords = getMostCommonWords(trainingData,vector_size)
# trainSentences = list(oneHotEncode(trainingData,mostCommonWords).values())
# valSentences = list(oneHotEncode)
# testData, spamTest = loadSMS2('SMSTest.txt')
# spamTest = convertSpamToBinary(spamTest)
# testSentences = sentenceEmbedding(testData,spamTest,embeddingDict,100)
# testSentences = list(oneHotEncode(testData,mostCommonWords).values())
# network = NeuralNetwork(2)
# network.train_network(10,trainSentences,spamData,0.00002,10)
# network.train_network(20,trainSentences,spamData,0.0004,9)
# print(network.test_network(testSentences,spamTest))