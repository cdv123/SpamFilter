import csv
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
def word_stemmer(words):
    stem_words = [stemmer.stem(o) for o in words]
    return " ".join(stem_words)
def word_lemmatizer(words):
   lemma_words = [lemmatizer.lemmatize(o) for o in words]
   return " ".join(lemma_words)
trainingData = []
stop_words = set(stopwords.words('english'))



#generalise function into load data
def loadTrainingData():
    punc = '''!()-[]{};:'"\,<>./?@#$%^&*_~0123456789'''
    stop_words = set(stopwords.words('english'))
    with open("./Dataset/training-data.csv") as file:
        trainingCSV= csv.reader(file)
        for row in trainingCSV:
            #Convert to lower case
            row[1] = row[1].lower()
            #Remove stop words
            row_tokens = word_tokenize(row[1])
            new_row = []
            for token in row_tokens:
                if token not in stop_words:
                    new_row.append(token)
            new_row = ' '.join(new_row)
            for char in new_row:
                if char in punc:
                    new_row = new_row.replace(char, "")
            new_row = word_stemmer(new_row.split(" "))
            new_row= word_lemmatizer(new_row.split(" "))
            trainingData.append(new_row)
    return trainingData
def loadTestData():
    testData= []
    punc = '''!()-[]{};:'"\,<>./?@#$%^&*_~0123456789'''
    stop_words = set(stopwords.words('english'))
    with open("./Dataset/test-data.csv") as file:
        stop_words = set(stopwords.words('english'))
        testingCSV= csv.reader(file)
        for row in testingCSV:
            #Convert to lower case
            row[1] = row[1].lower()
            #Remove stop words
            row_tokens = word_tokenize(row[1])
            new_row = []
            for token in row_tokens:
                if token not in stop_words:
                    new_row.append(token)
            new_row = ' '.join(new_row)
            for char in new_row:
                if char in punc:
                    new_row = new_row.replace(char, "")
            new_row = word_stemmer(new_row.split(" "))
            new_row= word_lemmatizer(new_row.split(" "))
            testData.append(new_row)
    return testData
def loadSMS(link):
    spamCount = 0
    hamCount = 0
    testData = []
    punc = '''!()-[]{};:'"\,<>./?@#$%^&*_~0123456789'''
    stop_words = set(stopwords.words('english'))
    spamList = []
    with open(f"./Dataset/{link}") as file:
        stop_words = set(stopwords.words('english'))
        testingCSV= csv.reader(file)
        for row in testingCSV:
            text =str(row[0][4:])
            #Convert to lower case
            text = text.lower()
            #Remove stop words
            row_tokens = word_tokenize(text)
            new_row = []
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
            spamList.append(row[0][:3])
            if spamList[-1] == 'spa':
                spamCount+=1
                spamList[-1] = 'spam'
            else:
                hamCount+=1
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
        testingCSV= csv.reader(file)
        for row in testingCSV:
            text =str(row[0][4:])
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
            spamList.append(row[0][:3])
            if spamList[-1] == 'spa':
                spamCount+=1
                spamList[-1] = 'spam'
            else:
                hamCount+=1
    return (testData,spamList)