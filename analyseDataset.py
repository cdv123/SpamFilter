import csv
# analysis on both training data and test data to make sure both have similar ratio of spam to ham
spamCountTest = 0
hamCountTest = 0
with open("./Dataset/spam_assassin.csv") as file:
    testData = csv.reader(file)
    for row in testData:
        if row[0] == 'ham':
            hamCountTest+=1
        else:
            spamCountTest+=1
