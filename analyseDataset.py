import csv
import pandas as pd
# analysis on both training data and test data to make sure both have similar ratio of spam to ham
# Testing
spamCountTest = 0
hamCountTest = 0
trainingCSV = pd.read_csv("Dataset/spam_assassin.csv")

for email in trainingCSV.index:
    pass
  
# writing into the file
trainingCSV.to_csv("Dataset/spam_assassin.csv", index=False)
        
