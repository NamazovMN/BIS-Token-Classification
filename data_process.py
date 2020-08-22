import numpy as np
import torch
import torch.nn as nn


class dataProcessing(object):
    def __init__(self, batch_size, size_of_vocab):
        self.batch_size = batch_size
        self.size_of_vocab = size_of_vocab
    
    def batch_generator(self, data_iterator, dataset_size):
        batched_data = {}
        batch_counter = 0
        counter = 0
        input_list = []
        target_list = []
        for each_input, each_target in data_iterator:
            counter +=1
            input_list.append(each_input)
            target_list.append(torch.FloatTensor(each_target))
            if (dataset_size-counter) < self.batch_size:
                target_tensor = torch.stack(target_list)
                # torch.cat(target_list, out = target_tensor)
                batched_data["sample " + str(batch_counter)] = {"input_data": input_list, "target_data": target_tensor}
            if counter % self.batch_size == 0:
                target_tensor = torch.stack(target_list)
                # torch.cat(target_list, out = target_tensor)
                batched_data["sample " + str(batch_counter)] = {"input_data": input_list, "target_data": target_tensor}
                batch_counter +=1
                input_list = []
                target_list = []
            
        return batched_data

    def set_one_hot(self, batch_sample):
        batch_one_hot = []
        for each in batch_sample:
            one_hot_vector = np.zeros(self.size_of_vocab)
            one_hot_vector[each] = 1
            one_hot_tensor = torch.FloatTensor(one_hot_vector)
            batch_one_hot.append(one_hot_tensor)
        tensor_new = torch.stack(batch_one_hot)
        return tensor_new

    def vocab_list(self, vocabulary):
        for each in vocabulary.keys():
            vocabulary[each] = list(vocabulary[each])
        return vocabulary



    def data_decode(self, vocabulary_dict, encoded_data):
        decoded_data = []
        for each_data in encoded_data:
            for each in vocabulary_dict.keys():
              if each_data == vocabulary_dict[each]:
            # if each_data[0] == vocabulary_dict[each][0] and each_data[1] == vocabulary_dict[each][1] and each_data[2] == vocabulary_dict[each][2]:
                decoded_data.append(each)
        return decoded_data

    def normalize_list(self, list_data):
        sum = 0
        for each in list_data:
            sum += each
        for each in list_data:
            each = each/sum
        return list_data

    def create_one_hot(self, predicted_data):
        result = []
        for predicted in predicted_data:
            predicted_norm = self.normalize_list(predicted)
            max_val = max(predicted_norm)
            idx = predicted_norm.index(max_val)
            one_hot_related = np.zeros(len(predicted_norm))
            one_hot_related[idx] = 1
            result.append(list(one_hot_related))
        return result


