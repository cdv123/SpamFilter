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
testData = []
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
