import torch
import torch.nn as nn
class Model_Network(nn.Module):

  def __init__(self, input_dimension, hidden_layer_dim, num_of_classes):
    super(Model_Network, self).__init__()
    self.input_dimension = input_dimension
    self.hidden_layer_dimension = hidden_layer_dim
    self.number_of_classes = num_of_classes

    self.fullyconnected1 = nn.Linear(self.input_dimension, self.hidden_layer_dimension)
    self.fullyconnected2 = nn.Linear(self.hidden_layer_dimension, self.number_of_classes)
    self.ReLU = nn.ReLU()
    self.sigmoid = nn.Sigmoid()
    self.loss_function = nn.MSELoss()


  def forward(self, input_data):
    out_first_layer = self.fullyconnected1(input_data)
    activated1 = self.ReLU(out_first_layer)
    out_second_layer = self.fullyconnected2(activated1)
    output = self.sigmoid(out_second_layer)
    return output