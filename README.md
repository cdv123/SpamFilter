# SpamFilter

---

Implementation of different machine learning techniques for evaluating whether a message is spam or not. After training, a graph of validation loss against training loss is shown as well.

## Different Models Trained

---

- Naive Bayes - counts the frequencies of each word in the training dataset where it is ham (not spam) vs when it is spam divided by size of vocabulary, using this it multiplies the probabilities of each word in the message and guesses spam if probability of spam is higher than probability of ham, else it assumes that it is ham
- The next 3 all are different ways of making the input words into vectors, which are then used to train a logistic regression binary classifier (one or two layers, can be configured)
  - One-hot encoding
  - Gensim Word2Vec - trained word2vec model on dataset using the gensim library and saved dictionary as a text-file "customEmbedding.txt" - sentence embeddings add made by averaging out the word embeddings of each word in the sentence
  - Skip-gram model - implemented the skip-gram model, settings such as word embedding dimension can be reconfigured - sentence embeddings add made by averaging out the word embeddings of each word in the sentence

## Link to Dataset Used

https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset
