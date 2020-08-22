class Data_Prepare_for_Test(object):
    def __init__(self, reader, vocabulary, vocab_dict):
        self.reader = reader
        self.vocabulary = vocabulary
        self.dictionary_vocab = vocab_dict
    # def set_dictionary_for_vocab(self, vocabulary):
    #     dictionary_vocab = {}
        
    #     for index in range(len(vocabulary)):
    #         dictionary_vocab[vocabulary[index]] = index
    #         print(dictionary_vocab)
    #         return dictionary_vocab

    def set_data_encoded(self):
        test_dataset_encoded = []
        print(self.dictionary_vocab.keys())
        for each_line in self.reader:
          for each_char in range(len(each_line)-1):
            if each_line[each_char] not in self.dictionary_vocab.keys():
              test_dataset_encoded.append(self.dictionary_vocab["<unk>"])
            else:
              test_dataset_encoded.append(self.dictionary_vocab[each_line[each_char]])
        return test_dataset_encoded
