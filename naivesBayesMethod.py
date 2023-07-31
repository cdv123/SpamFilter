from simplifyDataset import loadTrainingData
from collections import Counter
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
trainingData = loadTrainingData()
mostCommonWords = []
allTrainingWords = []
for row in trainingData:
    rowWords = row.split(" ")
    for i in rowWords:
        if i !='':
            allTrainingWords.append(i)
word_count = Counter(allTrainingWords)
word_count = word_count.most_common(1000)
for i in word_count:
    mostCommonWords.append()
print(word_count.most_common(1000))