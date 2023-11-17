import sys
import os

# Get the parent directory
parent_dir = os.path.dirname(os.path.realpath(__file__))[:-5]

# Add the parent directory to sys.path
sys.path.append(parent_dir)

from neuralNetwork import NeuralNetwork
from naiveBayesMethod import trainModel, useModel
from wordEmbedding import skip_gram_train
from simplifyDataset import loadSMS2, loadMessage
from naiveBayesMethod import trainModel
from oneHot import oneHotEncode, getMostCommonWords
from word2vec import useEmbedding2, sentenceEmbedding

# checks if data is of right length and labels of right format (1 or 0)

def test_loading(training_length, validation_length, testing_length):

    # load training data
    training_data, training_labels = loadSMS2('SMSSpamCollection.txt')
    # check if training data loaded properly
    check_data(training_data,training_labels, training_length)

    # load validation data
    val_data, val_labels= loadSMS2('SMSVal.txt')
    # check if validation data loaded properly
    check_data(val_labels, val_labels, validation_length)

    # load testing data
    test_data, test_labels = loadSMS2('SMSTest.txt')
    # check if testing data loaded properly
    check_data(test_data, test_labels, testing_length)

    messages = (training_data, val_data, test_data)
    labels = (training_labels, val_labels, test_labels)

    return messages, labels

def check_data(messages, labels, data_length):
    
    assert len(messages) == len(labels), "each spam message should have a label => should be of equal length"

    assert len(messages) == data_length, "there should be as many spam messages as the length of number of lines in the file"

    assert check_labels(labels), "all labels should either be 1 (ham) or 0 (spam)"

def check_one_hot(messages, dim):

    train_msgs, val_msgs, test_msgs = messages

    one_hot_train = check_dataset_one_hot(train_msgs, dim)
    one_hot_val = check_dataset_one_hot(val_msgs, dim)
    one_hot_test = check_dataset_one_hot(test_msgs, dim)
    
    return one_hot_train, one_hot_val, one_hot_test

def check_dataset_one_hot(msgs, dim):

    most_common_words = getMostCommonWords(msgs, dim)

    assert len(most_common_words) == dim, "dictionary of most common words equals to dimension"

    one_hot_msgs = oneHotEncode(msgs, most_common_words)

    # check vectors are of the right dimension
    right_dim = True
    for vector in one_hot_msgs:
        if len(vector) != dim:
            right_dim = False

    assert right_dim, "dimension of vector should be of the specified dimension"

    assert len(one_hot_msgs) == len(msgs), "there should be the same number of one hot encoded msgs as input msgs"

    correctness = True
    for i, vector in enumerate(one_hot_msgs):
        new_msg = [most_common_words[index] for index, val in enumerate(vector) if val == 1]
        msgs[i].split(" ")
        for word in new_msg:
            if word not in msgs[i]:
                correctness = False

    assert correctness, "one hot encoded words should be subsets of messages"

    return one_hot_msgs

def check_labels(spam_data):

    # since this is a binary classification task, labels should be either 1 or 0

    for i in spam_data:
        if i != 0 and i != 1:
            return False
        
    return True

def check_naive_bayes(messages, labels, word_num):

    # get probability of the word_num most common words of being spam or ham
    train_msgs, val_msgs, test_msgs = messages
    train_labels, val_labels, test_labels = labels

    word_prob_ham, word_prob_spam = trainModel(train_msgs, train_labels, word_num)

    assert len(word_prob_ham) == len(word_prob_spam) == word_num, "length of dictionary should equal word_num"

    assert word_prob_ham.keys() == word_prob_spam.keys(), "both dictionaries should have the same keys as each word has a probability of being spam and ham"

    # check all values in dictionary are between 0 and 1

    is_prob_bool = True

    for word in word_prob_spam:
        if word_prob_spam[word] < 0 or word_prob_spam[word] > 1 or word_prob_ham[word] < 0 or word_prob_ham[word] > 1:
            is_prob_bool == False

    assert is_prob_bool, "all values in both dictionaries should represent probabilities and hence be between 0 and 1"

    accuracy = useModel(test_msgs, test_labels, word_prob_ham, word_prob_spam)
    
    assert accuracy > 0.90, "accuracy should be at least 90%"

def check_word2vec(msgs):

    dim = 100
    embedding_dict = useEmbedding2()
    for word in embedding_dict:
        assert len(embedding_dict[word]) == dim, f"each embedding should have dimension specified, embedding for {word} incorrect"

    train_msgs, val_msgs, test_msgs = msgs
    train_embeddings = sentenceEmbedding(train_msgs, embedding_dict, dim)
    assert len(train_embeddings) == len(train_msgs)

    val_embeddings = sentenceEmbedding(val_msgs, embedding_dict, dim)
    assert len(val_embeddings) == len(val_msgs)

    test_embeddings = sentenceEmbedding(test_msgs, embedding_dict, dim)
    assert len(test_embeddings) == len(test_msgs)

    return train_embeddings, val_embeddings, test_embeddings

def check_skip_gram(messages, labels, epochs, dim, lrs):

    test_msgs = messages[-1]
    test_labels = labels[-1]
    messages = messages[:2]
    labels = labels[:2]

    # use word embeddings for spam classification 
    model, embedding = skip_gram_train(messages, labels, epochs, dim, lrs)

    # test model
    for word in embedding:
        assert len(embedding[word]) == dim, f"word embeddings created should be of specified dimension, embedding for {word} incorrect"

    # get sentence embeddings for testing data
    test_sentences = sentenceEmbedding(test_msgs, embedding, dim)
    accuracy = model.test_network(test_sentences,test_labels)   
    print("the accuracy of the model is:", accuracy) 
    assert accuracy > 0.9, "accuracy of model should be at least 90%"

# def check_one_layer(messages, labels, dims, lrs, embeddings):
#     pass

# def check_two_layer(messages, labels, dims, lrs):
#     pass

if __name__ == "__main__":

    training_length = 3498
    validation_length = 1000
    testing_length = 1006
    word_num = 1000
    dim = 300
    dim_sg = 60
    epochs_sg = (2, 20)
    lr_sg = (0.000003, 0.03)

    # test that data is loaded properly
    messages, labels = test_loading(training_length,validation_length,testing_length)

    # test naive bayes model
    check_naive_bayes(messages,labels,word_num)

    # test one hot encoding works as intended
    one_hot_train, one_hot_val, one_hot_test = check_one_hot(messages, dim)

    train_embeddings, val_embeddings, test_embeddings = check_word2vec(messages)

    check_skip_gram(messages, labels, epochs_sg, dim_sg, lr_sg)

    train_msgs, val_msgs, test_msgs = messages
    train_labels, val_labels, test_labels = labels

    print("Passed all tests!")