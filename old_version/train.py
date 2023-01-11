import torch
import torch.nn as nn
import numpy as np

class Training_Class(object):
  def __init__(self, model, loss_function, optimizer):
    self.model = model
    self.loss_function = loss_function
    self.optimizer = optimizer
    
  def set_one_hot(self, batch_sample, vocabulary_size):
    batch_one_hot = []
    for each in batch_sample:
        one_hot_vector = np.zeros(vocabulary_size)
        one_hot_vector[each] = 1
        one_hot_tensor = torch.FloatTensor(one_hot_vector)
        batch_one_hot.append(one_hot_tensor)
    tensor_new = torch.stack(batch_one_hot)
    return tensor_new

  def train(self, train_data, validation_data, epochs, vocabulary_size, batch_size):

    train_loss = 0
    for each_epoch in range(epochs):
      self.model.train()
      epoch_loss = 0

      step = 0
      for each_data in train_data: # step size considered as batch number 
        input_for_net_list = train_data[each_data]["input_data"]
        input_for_net = self.set_one_hot(input_for_net_list, vocabulary_size).to("cuda")
        target_for_net = train_data[each_data]["target_data"].to("cuda")

        self.optimizer.zero_grad()
        output = self.model(input_for_net)
        batch_loss = self.loss_function(output, target_for_net)
        batch_loss.backward()
        self.optimizer.step()

        epoch_loss += batch_loss.tolist()
        

      avg_epoch_loss = epoch_loss/len(train_data)

      train_loss += avg_epoch_loss

      validation_loss = self.validation_phase(validation_data, batch_size, vocabulary_size)
      print('Epoch: {} Train Loss: {:0.4f} Validation Loss: {:0.4f} '.format(each_epoch+1, avg_epoch_loss, validation_loss ))


      # print(" Epoch :   {:2d}    Average validation accuracy :    {:0.6f}   Train loss: {:0.4f}".format(each_epoch + 1, 1-validation_loss, train_loss/epochs))
    
    return train_loss / epochs

  def validation_phase(self, valid_data, batch_size, vocabulary_size):
    valid_loss = 0

    with torch.no_grad():
      for each_dict in valid_data:
        input_for_valid_list = valid_data[each_dict]["input_data"]

        target_for_valid = valid_data[each_dict]["target_data"].to("cuda")
        input_for_valid = self.set_one_hot(input_for_valid_list,vocabulary_size).to("cuda")

        out_valid = self.model(input_for_valid)
        batch_loss_valid = self.loss_function(out_valid, target_for_valid)
        valid_loss += batch_loss_valid.tolist()
    
    return valid_loss/len(valid_data)


  def predict(self, input):
    return self.model(input).tolist()
