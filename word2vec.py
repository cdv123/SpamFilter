import numpy as np
def sentenceEmbedding(data,spamList,word2vec,vector_size):
    sentenceEmbeddings = []
    dataLength = len(data)
    i = 0
    while i < dataLength:
        sentence = np.zeros(vector_size)
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
def tokenize(trainingData):
    wordEmbedding = []
    for i in range(len(trainingData)):
        row = trainingData[i].split(" ")
        newRow = []
        for word in row:
            if word != '':
                newRow.append(word)
        if newRow != []:
            wordEmbedding.append(newRow)
    return wordEmbedding

def sentenceEmbedding2(data,spamList,word2vec,vector_size):
    sentenceEmbeddings = []
    dataLength = len(data)
    i = 0
    while i < dataLength:
        sentence = np.zeros(vector_size)
        row = data[i].split(" ")
        length = 0
        for j in row:
            try:
                sentence = np.add(sentence,np.array(word2vec.wv[j]))
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

