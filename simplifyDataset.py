# from nltk.corpus import stopwords
# def word_stemmer(words):
#     stem_words = [stemmer.stem(o) for o in words]
#     return " ".join(stem_words)
# def word_lemmatizer(words):
#    lemma_words = [lemmatizer.lemmatize(o) for o in words]
#    return " ".join(lemma_words)
stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]
def loadSMS(data):
    spamCount = 0
    hamCount = 0
    testData = []
    stop_words = set(stopwords)
    punc = '''!()-[]{};:'"\,<>./?@#$%^&*_~0123456789£'''
    datalines = data.split("\n")
    spamList = []
    for row in datalines:
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
    spamList = []
    with open(f"./Dataset/{link}") as file:
        stop_words = set(stopwords)
        testingCSV= file.readlines()
        for row in testingCSV:
            text =str(row[4:])
            #Convert to lower case
            text = text.lower()
            #Remove stop words
            # row_tokens = word_tokenize(text)
            new_row = []
            row_tokens = text.split()
            for token in row_tokens:
                if token not in stop_words:
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
stop_words = set(stopwords)
punc = '''!()-[]{};:'"\,<>./?@#$%^&*_~0123456789'''
def loadMessage(message):
    message = message.lower()
    new_message =[]
    message_tokens = message.split(' ')
    for token in message_tokens:
        if token not in stop_words:
            new_message.append(token)
    new_message = ' '.join(new_message)
    for char in new_message:
        if char in punc:
            new_message = new_message.replace(char,"")
    return new_message
def convertSpamToBinary(spamList):
    for i in range(len(spamList)):
        if spamList[i] == 'ham':
            spamList[i] = 0
        else:
            spamList[i] = 1
    return spamList