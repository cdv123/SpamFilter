# from nltk.corpus import stopwords
# def word_stemmer(words):
#     stem_words = [stemmer.stem(o) for o in words]
#     return " ".join(stem_words)
# def word_lemmatizer(words):
#    lemma_words = [lemmatizer.lemmatize(o) for o in words]
#    return " ".join(lemma_words)
def loadSMS(data,stopwords):
    spamCount = 0
    hamCount = 0
    testData = []
    stop_words = set(stopwords)
    punc = '''!()-[]{};:'"\,<>./?@#$%^&*_~0123456789'''

    spamList = []
    for row in data:
        text =str(row[4:])
        #Convert to lower case
        text = text.lower()
        #Remove stop words
        for char in text:
            if char in punc:
                text = text.replace(char, "")
        row_tokens = text.split()
        new_row = []
        for token in row_tokens:
            if token not in stop_words:
                new_row.append(token)
        new_row = ' '.join(new_row)
        # new_row = word_stemmer(new_row.split(" "))
        # new_row= word_lemmatizer(new_row.split(" "))
        testData.append(new_row)
        spamList.append(row[:4])
        if spamList[-1] == '"spa':
            spamCount+=1
            spamList[-1] = 'spam'
        else:
            hamCount+=1
            spamList[-1] = 'ham'
    return (testData,spamList)
def loadSMS2(link):
    spamCount = 0
    hamCount = 0
    testData = []
    punc = '''!()-[]{};:'"\,<>./?@#$%^&*_~0123456789'''
    stop_words = set(stopwords.words('english'))
    spamList = []
    with open(f"./Dataset/{link}") as file:
        stop_words = set(stopwords.words('english'))
        testingCSV= file.readlines()
        for row in testingCSV:
            text =str(row[4:])
            #Convert to lower case
            text = text.lower()
            #Remove stop words
            row_tokens = word_tokenize(text)
            new_row = []
            for token in row_tokens:
                new_row.append(token)
            new_row = ' '.join(new_row)
            for char in new_row:
                if char in punc:
                    new_row = new_row.replace(char, "")
            # new_row = word_stemmer(new_row.split(" "))
            # new_row= word_lemmatizer(new_row.split(" "))
            testData.append(new_row)
            spamList.append(row[:4])
            if spamList[-1] == '"spa':
                spamCount+=1
                spamList[-1] = 'spam'
            else:
                hamCount+=1
                spamList[-1] = 'ham'
                
    return (testData,spamList)
def convertSpamToBinary(spamList):
    for i in range(len(spamList)):
        if spamList[i] == 'ham':
            spamList[i] = 0
        else:
            spamList[i] = 1
    return spamList
