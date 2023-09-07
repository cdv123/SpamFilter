from collections import Counter
from simplifyDataset import loadMessage
def trainModel(trainingData,spamList):    
    mostCommonWords = []
    allTrainingWords = []
    spamWordsList = []
    hamWordsList = []
    spamCount = 0
    length = len(trainingData)
    hamCount = 0
    for row in range(length):
        rowWords = trainingData[row].split(" ")
        if spamList[row] == 1:
            for i in rowWords:
                if i !='':
                    spamCount+=1
                    spamWordsList.append(i)
        else:
            for i in rowWords:
                if i !='':
                    hamCount+=1
                    hamWordsList.append(i)
    allTrainingWords = hamWordsList+spamWordsList
    vocab = set(allTrainingWords)
    word_count = Counter(allTrainingWords)
    word_count = word_count.most_common(1000)
    mostCommonWords = []
    for i in range(0,len(word_count)):
        mostCommonWords.append(word_count[i][0])
    wordProbSpam = {}
    wordProbHam = {}
    for i in mostCommonWords:
        wordProbSpam[i] = (spamWordsList.count(i))/(spamCount+len(vocab))
        wordProbHam[i] = (hamWordsList.count(i))/(hamCount+len(vocab))
    return wordProbHam,wordProbSpam
def useModel(testData,spamTestList,wordProbHam,wordProbSpam):
    correctCount =0
    incorrectCount = 0
    isSpam = True
    for i in range(len(testData)):
        hamProb = 0.85
        spamProb = 0.15
        row = testData[i].split(" ")
        for j in row:
            if j != '':
                if wordProbHam.get(j) == None or wordProbHam.get(j) == 0:  
                    wordProbHam[j] = 1/2
                    wordProbSpam[j] = 1/2
                hamProb*=wordProbHam[j]
                spamProb*= wordProbSpam[j]
        if hamProb > spamProb:
            isSpam = False
        else:
            isSpam = True
        if (isSpam == True and spamTestList[i] == 1) or (isSpam == False and spamTestList[i] == 0):
            correctCount+=1
        else:
            incorrectCount+=1
    accuracy = correctCount/(incorrectCount+correctCount)
    return accuracy
def analyseMsg(wordProbHam,wordProbSpam,message):
    message = loadMessage(message)
    message_words = message.split(' ')
    hamProb = 0.85
    spamProb = 0.15
    for i in range(len(message_words)):
        if message_words[i]!= ' ':
            if wordProbHam.get(message_words[i]) == None or wordProbHam.get(message_words[i]) == 0:  
                wordProbHam[message_words[i]] = 1/2
                wordProbSpam[message_words[i]] = 1/2
            hamProb*=wordProbHam[message_words[i]]
            spamProb*= wordProbSpam[message_words[i]]
    spamProb = spamProb/(spamProb+hamProb)
    return spamProb