import os
import sys
import torch
import torch.nn as nn
from data_preparation import Data_Preparation
from data_reader import Data_Reader
from model import Model_Network
from data_for_test import Data_Prepare_for_Test
from prep_for_model import Prepare_Data_For_Model
from train import Training_Class
from data_process import dataProcessing


dataset_path = "Dataset"

for i in range(2):
    if i ==0:
        train_input = "en.wiki.sentences.train"
        train_input_reader = Data_Reader(dataset_path, train_input)

        train_target = "en.wiki.gold.train"
        train_target_reader = Data_Reader(dataset_path, train_target)

        validation_input = "en.wiki.sentences.dev"
        validation_input_reader = Data_Reader(dataset_path,validation_input)

        validation_target = "en.wiki.gold.dev"
        validation_target_reader = Data_Reader(dataset_path, validation_target)

        path_for_model = os.path.join(dataset_path, "saved_model")

        test_reader = Data_Reader(dataset_path, "en.wiki.sentences.test")

    else:
        train_input = "en.wiki.merged.sentences.train"
        train_input_reader = Data_Reader(dataset_path, train_input)

        train_target = "en.wiki.merged.gold.train"
        train_target_reader = Data_Reader(dataset_path, train_target)

        validation_input = "en.wiki.merged.sentences.dev"
        validation_input_reader = Data_Reader(dataset_path,validation_input)

        validation_target = "en.wiki.merged.gold.dev"
        validation_target_reader = Data_Reader(dataset_path, validation_target)

        path_for_model = os.path.join(dataset_path, "saved_model_merged")

        test_reader = Data_Reader(dataset_path, "en.wiki.merged.sentences.test")

    data_preparator = Data_Preparation(train_input_reader, train_target_reader, validation_input_reader)
    vocabulary_size = data_preparator.vocab_size
    num_of_classes = data_preparator.num_of_classes

    hidden_layer_dim = 60
    learning_rate = 0.05
    batch_size = 256
    epoch_num = 30

    data_for_NN = Prepare_Data_For_Model(data_preparator, train_input_reader, train_target_reader, validation_input_reader, validation_target_reader)
    train_iter = data_for_NN.set_input_data(train = True)
    validation_iterator = data_for_NN.set_input_data(train = False)
    vocab_dict = data_preparator.vocabulary_dictionary
    train_data_size = data_preparator.counter_train_input
    validation_data_size = data_preparator.counter_valid_input

    data_processing = dataProcessing(batch_size, vocabulary_size)

    train_dataset = data_processing.batch_generator(train_iter, train_data_size)
    validation_dataset = data_processing.batch_generator(validation_iterator, validation_data_size)

    model = Model_Network(vocabulary_size, hidden_layer_dim, num_of_classes).to("cuda")
    optimizer = torch.optim.SGD(model.parameters(), learning_rate)
    loss_function = nn.MSELoss()

    train_phase = Training_Class(model, loss_function, optimizer)
    if i == 0:
        print("Training has started for normal sentences!")
    else:
        print("Training has started for merged sentences!")
    avg_loss = train_phase.train(train_dataset, validation_dataset, epoch_num, vocabulary_size, batch_size)


    torch.save(model.state_dict(), path_for_model)
    saved =1

    if saved == 1:
      model_new = Model_Network(vocabulary_size, hidden_layer_dim, num_of_classes)
      model_new.load_state_dict(torch.load(path_for_model))
      model_new.eval()
    else:
      print("There is not any saved model, yet. You should train the network first!")


    test_preparator = Data_Prepare_for_Test(test_reader, data_preparator.vocabulary, vocab_dict)
    test_dataset_list = test_preparator.set_data_encoded()
    test_data_one_hot = data_processing.set_one_hot(test_dataset_list)  

    vocab_dict_clasess = data_preparator.vocab_dict_classes

    test_predicted = train_phase.predict(test_data_one_hot)
    test_predicted = model_new(test_data_one_hot).tolist()

    test_one_hot_data = data_processing.create_one_hot(test_predicted)
    vocab_classes_list = data_processing.vocab_list(vocab_dict_clasess)

    decoded_data = data_processing.data_decode(vocab_classes_list, test_one_hot_data)

    print(len(decoded_data))
    count_char = 0

    if i ==0:
        file_new = open("en.wiki.predicted.test", "w+")
    else:
        file_new = open("en.wiki.merged.predicted.test", "w+")

    for each_line in test_reader:
      line = ''
      for i in range(len(each_line)-1):
          line += decoded_data[count_char]
          count_char +=1
      line += '\n' 
      file_new.write(line)
      file_new.close()

    saved = 0
