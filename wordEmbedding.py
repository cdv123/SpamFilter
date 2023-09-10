from collections import Counter
from simplifyDataset import loadSMS2
import numpy as np
import random
#Steps
#Read text
#Preprocess text
#Create (x,y) data points
#Create one hot encoded (X,Y) matrices
#train neural network
#neural network should have one input layer, one hidden layer and one output layer
#extract weights from hidden layer
import math
def get_text(training_data):
    text = []
    for row in training_data:
        row_words = row.split()
        for i in range(len(row_words)):
            text.append(row_words[i])
    return text
def make_all_pairs(training_data,window,text_set):
    word_pairs=[]
    for row in training_data:
        row_words = row.split()
        remove_words(row_words,text_set)
        for i in range(len(row_words)):
            pair_list = make_row_pairs(window,row_words,i)
            word_pairs+= pair_list
    return word_pairs
def remove_words(row_words,text_set):
    for i in row_words:
        if i not in text_set:
            row_words.remove(i)
def make_row_pairs(window,row_words,index):
    j = 1
    pair_list = []
    while index-j>=0 and j<=window:
        pair_list.append((row_words[index],row_words[index-j]))
        j+=1
    j = 1
    while index+j<len(row_words) and j<= window:
        pair_list.append((row_words[index],row_words[index+j]))
        j+=1
    return pair_list
def uniqueWordDict(text):
    text = list(set(text))
    word_dict = {}
    j=0
    for i in text:
        word_dict[i] = j
        j+=1
    return word_dict
def sampling_rate(frequency):
    # ignore words which appear too often as they provide little information
    prob = 0.001/frequency
    multiplier = math.sqrt(frequency/0.001)+1
    prob*=multiplier
    return prob
def make_matrices(word_dict,pairs):
    focus_matrix = []
    context_matrix = []
    for pair in pairs:
        focus_word_index = word_dict[pair[0]]
        context_word_index = word_dict[pair[1]]
        focus_matrix.append(focus_word_index)
        context_matrix.append(context_word_index)
    return focus_matrix,context_matrix

def skip_gram_model(focus_matrix,context_matrix,dim,epochs,lr,vector_size): 
    np.random.seed(0)
    limit= math.sqrt(6/(vector_size+dim))
    # weight matrix to go from input layer to hidden layer
    hidden_w = np.random.uniform(-limit,limit,size=(vector_size,dim))
    # hiddenW = np.zeros((vectorSize,hiddenNodesNum))
    limit=math.sqrt(6/(1+dim))
    # weight vector to go from hidden layer to output layer
    output_w = np.random.uniform(-limit,limit,size=(dim,vector_size))
    # outputW =np.zeros(hiddenNodesNum)
    # bias scalar to go from hidden layer to output layer
    output_bias = np.zeros(vector_size)
    for epoch in range(epochs):
        for x in range(len(focus_matrix)):
            # compute predictions of model
            hidden_res,final_res = forward_pass(focus_matrix[x],hidden_w,output_w,output_bias)
            # softmax used as multiple classes used
            final_res = softmax(final_res)
            loss = loss_function(final_res,context_matrix[x])
            error_der = softmax_der(final_res,context_matrix[x])
            gradb = error_der
            gradx = compute_gradient(hidden_res,gradb)
            output_w,output_bias = update_weights(output_w,output_bias,gradx,gradb,lr)
            # THIS NEEDS TO BE CHANGED
            hidden_loss = hidden_w*loss
            grad_hidden_w = compute_hidden_gradient(hidden_loss,focus_matrix[x]) 
            hidden_w = update_weights(hidden_w,0,grad_hidden_w,0,lr)[0]
    return hidden_w,output_w,output_bias
def forward_pass(x,hidden_w,output_w,output_bias):
    # since input vector only has a single 1, the output of the first matrix multiplication 
    # is the row corresponding to the index of this 1
    hidden_res = hidden_w[x]
    final_res = np.matmul(hidden_res,output_w)
    final_res += output_bias
    return hidden_res,final_res

def compute_hidden_gradient(errorHidden,vectorInput):
    gradHiddenW = errorHidden[vectorInput]
    return gradHiddenW

def compute_gradient(x,gradb):
    gradb = np.reshape(gradb,(1,-1))
    x = np.reshape(x,(-1,1))
    gradx = x*gradb
    return gradx

def softmax(x):
    y = np.exp(x)/np.sum(np.exp(x))
    return y

def loss_function(yhat,y):
    cost = -1*np.log(yhat[y])
    return cost

def softmax_der(yhat,y):
    yhat[y]-=1
    return yhat

def update_weights(w,b,dw,db,lr):
    w = w-dw*lr
    b = b-db*lr
    return w,b

def subsampling(prob):
    rnd = random.random()
    if prob>rnd:
        return True
    return False

def negative_sampling_prob():
    pass
training_data,spam_data = loadSMS2("SMSSpamCollection copy.txt")
text = get_text(training_data)
text_num = len(text)
text_set = set(text)
text_counter = Counter(text)
new_text = set()
for i in text_set:  
    if subsampling(sampling_rate(text_counter[i]/text_num)): 
        new_text.add(i)
pairs = make_all_pairs(training_data,5,text)
unique_word_dict = uniqueWordDict(text)
focus_matrix,context_matrix = make_matrices(unique_word_dict,pairs)
skip_gram_model(focus_matrix,context_matrix,100,10,0.1,len(unique_word_dict))
# print(len(new_text))
# text_counter = text_counter.most_common()
# print(text_counter)