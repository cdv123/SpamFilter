from collections import Counter

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
def oneHotEncode2(message,mostCommonWords):
    oneHot = [0]*len(mostCommonWords)
    messageWords = message.split(' ')
    for j in range(len(mostCommonWords)):
        if mostCommonWords[j] in messageWords:
            oneHot[j] = 1
    return oneHot
