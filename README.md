# SpamFilter
---
Implementation of different machine learning techniques for evaluating whether a message is spam or not. There is also a website which can be used to compare the evaluations of different models on a message, and these models can be reconfigured.

Website can be run by running a local http server as it is a static website.

## Different Models Trained
---
- Naive Bayes - counts the frequencies of each word in the training dataset where it is ham (not spam) vs when it is spam divided by size of vocabulary, using this it multiplies the probabilities of each word in the message and guesses spam if probability of spam is higher than probability of ham, else it assumes that it is ham
- The next 3 all are different ways of making the input words into vectors, which are then used to train a logistic regression binary classifier (one or two layers, can be configured)
  - One-hot encoding
  - Gensim Word2Vec - trained word2vec model on dataset using the gensim library and saved dictionary as a text-file "customEmbedding.txt" 
  - Skip-gram model - implemented the skip-gram model, settings such as word embedding dimension can be reconfigured.

  ## Link to Dataset Used
  https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset
