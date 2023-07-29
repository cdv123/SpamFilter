import csv
# analysis on both training data and test data to make sure both have similar ratio of spam to ham
spamCountTest = 0
hamCountTest = 0
with open("./Dataset/test-data.csv") as file:
    testData = csv.reader(file)
    for row in testData:
        if row[0] == 'ham':
            hamCountTest+=1
        else:
            spamCountTest+=1
print("Percentage of spam in test data:")
perSpamTest = spamCountTest/(hamCountTest+spamCountTest)*100
print(perSpamTest)
spamCountTraining = 0
hamCountTraining = 0
with open("./Dataset/training-data.csv") as file:
    testData = csv.reader(file)
    for row in testData:
        if row[0] == 'ham':
            hamCountTraining+=1
        else:
            spamCountTraining+=1
print("Percentage of spam in training data:")
perSpamTraining = spamCountTraining/(hamCountTraining+spamCountTraining)*100
print(perSpamTraining)

print("Percentage difference between amount of spam in test data and training data")
print((perSpamTest-perSpamTraining))

print("Value above as a percentage of spam in training data set")
print((perSpamTest-perSpamTraining)/perSpamTraining*100)