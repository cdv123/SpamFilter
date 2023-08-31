import numpy as np
def useEmbedding(file):
    datalines = file.split("\n")    
    embeddingDict = {}
    i = 0
    while i< len(datalines):
        word = datalines[i][:-1]
        i+=1
        stringNums = ""
        while i<len(datalines) and datalines[i][0] == " ":
            stringNums += datalines[i]
            i+=1
        stringArray = stringNums.split(' ')
        modArray =[i for i in stringArray if i!='' and i!='\n' and i !='\r']
        vector = np.array([float(j) for j in modArray if j !=''])
        embeddingDict[word] = vector
    return embeddingDict
def useEmbedding2():
    with open("gloveEmbedding.txt", encoding="utf8") as f:
        lines = f.readlines()
        embeddingDict = {}
        for line in lines:
            row = line.split(' ')
            word = row[0]
            vector = np.array([float(row[j]) for j in range(1,len(row))])
            embeddingDict[word] = vector
    return embeddingDict
#average out all word embeddings into sentence embedding
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