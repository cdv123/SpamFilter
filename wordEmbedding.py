from collections import Counter
from simplifyDataset import loadSMS2, convertSpamToBinary
import numpy as np
import random
from numpyNetwork import neuralNetwork,testNetwork,twoLayerNetwork,testTwoLayer
from word2vec import sentenceEmbedding
import cProfile

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

def make_matrices_2(word_dict,pairs):
    focus_matrix = []
    context_matrix = []
    prev_pair = pairs[0][0]
    pair = 0
    while pair < len(pairs):
        focus_matrix.append(word_dict[pairs[pair][0]])
        context_matrix.append([])
        while pair < len(pairs) and prev_pair == pairs[pair][0]:
            context_matrix[-1].append(word_dict[pairs[pair][1]])
            pair+=1
        if pair >= len(pairs):
            break
        prev_pair = pairs[pair][0]
    return focus_matrix,context_matrix
def skip_gram_model(focus_matrix,context_matrix,dim,epochs,lr,text,word_dict,n_words): 
    np.random.seed(0)
    random.seed(0)
    vector_size = len(word_dict)
    limit= math.sqrt(6/(vector_size+dim))
    text_size = len(text)
    # weight matrix to go from input layer to hidden layer
    # hidden_w = np.random.uniform(-limit,limit,size=(vector_size,dim))
    hidden_w = np.random.uniform(-limit,limit,size=(vector_size,dim))
    # hiddenW = np.zeros((vectorSize,hiddenNodesNum))
    limit=math.sqrt(6/(1+dim))
    # weight vector to go from hidden layer to output layer
    output_w = np.random.uniform(-limit,limit,size=(dim,vector_size))
    loss = 0
    text_size = len(text)
    # outputW =np.zeros(hiddenNodesNum)
    for epoch in range(epochs):
        for x in range(len(focus_matrix)):
            # compute predictions of model
            hidden_res,final_res = forward_pass(focus_matrix[x],hidden_w,output_w)
            # softmax used as multiple classes used
            final_res = softmax(final_res)
            # loss += loss_function(final_res,context_matrix[x])
            error_der = softmax_der(final_res,context_matrix[x])
            hidden_w,output_w = back_propagation(output_w,hidden_w,focus_matrix[x],context_matrix[x],hidden_res,error_der,lr,text_size,text,n_words,word_dict)
            # gradx = compute_gradient(hidden_res,error_der)
            # hidden_loss = hidden_w.transpose()*error_der
            # grad_hidden_w = compute_hidden_gradient(hidden_loss,focus_matrix[x]) 
            # output_w= update_weights(output_w,gradx,lr)
            # hidden_w = update_weights(hidden_w,grad_hidden_w,lr)
        loss = loss/(len(focus_matrix))
        print(epoch,loss)
        loss = 0
    return hidden_w
def forward_pass(x,hidden_w,output_w):
    # since input vector only has a single 1, the output of the first matrix multiplication 
    # is the row corresponding to the index of this 1
    hidden_res = hidden_w[x]
    final_res = np.dot(hidden_res,output_w)
    return hidden_res,final_res

def compute_hidden_gradient(hidden_loss,vector_input):
    grad_hidden_w = hidden_loss[:,vector_input]
    return grad_hidden_w

def compute_gradient(x,error_der):
    x = np.reshape(x,(-1,1))
    gradx = error_der*x
    return gradx

def softmax(x):
    y = np.exp(x)/np.sum(np.exp(x))
    return y

def loss_function(yhat,y):
    cost = -1*np.log(yhat[y])
    return cost

def softmax_der(yhat,y):
    for val in y:
        yhat[val] -=1
    return yhat

def update_weights(w,dw,lr):
    w = w-dw*lr
    return w

def back_propagation(output_w,hidden_w,input,output,hidden_res,error,lr,text_size,text_words,n_words,word_dict):
    words_to_change = choose_samples(text_size,text_words,n_words,word_dict)
    grad_output_w = [error[words_to_change[i]]*hidden_res if i < len(words_to_change) else error[output[i-len(words_to_change)]]*hidden_res for i in range(len(words_to_change)+len(output))]
    # grad_output_w = np.zeros((len(words_to_change)+len(output),len(hidden_res)))
    # for i in range(len(words_to_change)):
    #     grad_output_w[i] = error[words_to_change[i]]*hidden_res
    # for i in range(len(output)):
    #     grad_output_w[len(words_to_change)+i] = error[output[i]]*hidden_res
    grad_hidden_w = np.dot(output_w,error.T)
    for i in range(len(words_to_change)):
        output_w[:,words_to_change[i]] = output_w[:,words_to_change[i]] - grad_output_w[i]*lr
    for i in range(len(output)):
        output_w[:,output[i]] = output_w[:,output[i]]-grad_output_w[len(words_to_change)+i]
    hidden_w[input] = hidden_w[input]-lr*grad_hidden_w
    return hidden_w,output_w

def subsampling(prob):
    rnd = random.random()
    if prob>rnd:
        return True
    return False

def choose_samples(text_size,text_words,n_words,word_dict):
    samples = [word_dict[text_words[random.randint(0,text_size-1)]] for i in range(n_words)]
    return samples

def make_mini_batches(focus_matrix,context_matrix,batch_size):
    mini_batches = []
    i = 0
    while i<len(focus_matrix):
        if i+batch_size<len(focus_matrix):
            mini_batch_focus = [focus_matrix[i+j] for j in range(batch_size)]
            mini_batch_context = [context_matrix[i+j] for j in range(batch_size)]
            mini_batches.append((mini_batch_focus,mini_batch_context))
        else:
            mini_batch_focus = [focus_matrix[i+j] for j in range(len(focus_matrix)-i)]
            mini_batch_context = [context_matrix[i+j] for j in range(len(focus_matrix)-i)]
            mini_batches.append((mini_batch_focus,mini_batch_context))
        i+=batch_size
    return mini_batches

def preprocess_text(training_data):
    text = get_text(training_data)
    text_num = len(text)
    text_set = set(text)
    text_counter = Counter(text)
    new_text = set()
    for i in text_set:  
        if subsampling(sampling_rate(text_counter[i]/text_num)): 
            new_text.add(i)
    pairs = make_all_pairs(training_data,2,text)
    unique_word_dict = uniqueWordDict(text)
    focus_matrix,context_matrix = make_matrices_2(unique_word_dict,pairs)
    return new_text,unique_word_dict,focus_matrix,context_matrix
# training_data,spam_data = loadSMS2("SMSSpamCollection.txt")s
# new_text,unique_word_dict,focus_matrix,context_matrix = preprocess_text(training_data)
# mini_batches = make_mini_batches(focus_matrix,context_matrix,32)
# hidden_w = skip_gram_model(focus_matrix,context_matrix,50,3,0.000015,list(new_text),unique_word_dict,20)
# hidden_w = skip_gram_model(focus_matrix,context_matrix,50,3,0.0004,list(new_text),unique_word_dict,20)
# cProfile.run('skip_gram_model(focus_matrix,context_matrix,50,3,0.000015,list(new_text),unique_word_dict,20)')
# word_embedding = {}
# for i in unique_word_dict:
#     word_embedding[i] = hidden_w[unique_word_dict[i]]
# # training_data2,spam_data2 = loadSMS2('SMSSpamCollection.txt')
# spam_data = convertSpamToBinary(spam_data)
# vector_size = 50    
# trainSentences = sentenceEmbedding(training_data,spam_data,word_embedding,50)
# mostCommonWords = getMostCommonWords(trainingData,vector_size)
# oneHotTrain = list(oneHotEncode(trainingData,mostCommonWords).values())
# weights,bias,hiddenW,hiddenBias = neuralNetwork(trainSentences,spam_data,0.004,9)
# hidden_weights,hidden_bias,output_weights,output_bias = twoLayerNetwork(trainSentences,spam_data,0.0002,9,20)
# print(bias)
# testData, spamTest = loadSMS2('SMSTest.txt')
# # oneHotTest = list(oneHotEncode(testData,mostCommonWords).values())
# spamTest = convertSpamToBinary(spamTest)
# testSentences = sentenceEmbedding(testData,spamTest,word_embedding,50)
# # print(testNetwork(testSentences,spamTest,weights,bias))
# print(testTwoLayer(testSentences,spamTest,hidden_weights,hidden_bias,output_weights,output_bias))

