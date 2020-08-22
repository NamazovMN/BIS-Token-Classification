import os
import numpy as np
class Data_Preparation(object):
  def __init__(self, train_x_reader, train_y_reader, validation_x_reader, lower = False):
    self.train_x_reader = train_x_reader
    self.train_y_reader = train_y_reader
    self.validation_x_reader = validation_x_reader
    self.lower = lower
    self.vocabulary_create()
    # classes are encoding according to one-hot encoding sturcture

  def vocabulary_create(self):
    if self.lower:
      print("According to the demand, lowercase characters are gathered!")
    self.vocabulary_train_input, self.counter_train_input = self.get_vocabulary_wrt_lower(self.train_x_reader)
    self.vocabulary_target, _ = self.get_vocabulary_wrt_lower(self.train_y_reader)
    self.vocabulary_valid_input, self.counter_valid_input = self.get_vocabulary_wrt_lower(self.validation_x_reader)
    self.vocabulary = self.merge_vocabs(self.vocabulary_train_input, self.vocabulary_valid_input) 
    self.vocab_size = len(self.vocabulary)
    self.vocabulary_dictionary = self.set_index_encoding(self.vocab_size)
    self.num_of_classes = len(self.vocabulary_target)
    self.vocab_dict_classes = self.set_one_hot_encoding(self.num_of_classes)
  
  def get_vocabulary_wrt_lower(self, reader):
    count_all_chars = 0
    vocabulary = []
    

    for each_line in reader:
      for i in range(len(each_line)-1):
        count_all_chars +=1
        if self.lower:      
          char_lower = each_line[i].lower()
          if char_lower not in vocabulary:
            vocabulary.append(char_lower)
        else:
          if each_line[i] not in vocabulary:
            vocabulary.append(each_line[i])
    counter = count_all_chars
    
    print("Vocabulary of chars is : {0} \nThere are {1} different characters in vocabulary.\nHowever there are {2} characters in total".format(vocabulary, len(vocabulary), count_all_chars))
    return vocabulary, counter

  def merge_vocabs(self, train_vocab, validation_vocab):
    vocabulary = train_vocab
    for each in validation_vocab:
      if each not in train_vocab:
        vocabulary.append(each)
    vocabulary.append("<unk>")
    return vocabulary
  # this method generates vocabulary dictionary it has vectors as much as vocabulary (set of all unique chars) vectors in that dimension
  # if there 455 vectors, each vector has 455x1 dimension
  # key - char itself, corresponding value - one hot vector, data type is np.array
  def set_one_hot_encoding(self, length):
    dictionary_vocab = {}

    for index in range(length):
      vector = np.zeros(length)
      vector[index] = 1
      dictionary_vocab[self.vocabulary_target[index]] = np.array(vector)
      
    return dictionary_vocab #structure (ith key (it is char): ith val (it is np.array type one hot vector))

  def set_index_encoding(self, length):
    dictionary_vocab = {}

    for index in range(length):
      dictionary_vocab[self.vocabulary[index]] = index
    print(dictionary_vocab)
    return dictionary_vocab

  def set_all_data_encoded(self, reader, target = True):
    # by using the following line, we get our vocabulary
    if target:
      dict_vocab = self.set_one_hot_encoding(self.num_of_classes)
    else:
      dict_vocab = self.set_index_encoding(self.vocab_size)

    for line in reader:
      for i in range(len(line)-1):
        yield dict_vocab[line[i]]