

class Prepare_Data_For_Model(object):

  def __init__(self, data_preparator, train_input_reader, train_target_reader, validation_input_reader, validation_target_reader):
    self.data_preparator = data_preparator
    self.train_input_reader = train_input_reader
    self.train_target_reader = train_target_reader
    self.validation_input_reader = validation_input_reader
    self.validation_target_reader = validation_target_reader

  def set_input_data(self, train = True):
    
    train_input_iter = self.data_preparator.set_all_data_encoded(self.train_input_reader, target = False) #iterator through x data of train 
    train_target_iter = self.data_preparator.set_all_data_encoded(self.train_target_reader) #iterator through y data of train 
    valid_input_iter = self.data_preparator.set_all_data_encoded(self.validation_input_reader, target = False) #iterator through x data of test 
    valid_target_iter = self.data_preparator.set_all_data_encoded(self.validation_target_reader) #iterator through y data of test 


    if train:
      for each_x, each_y in zip(train_input_iter, train_target_iter):
        yield each_x, each_y
    else:
      for each_x_valid, each_y_valid in zip(valid_input_iter, valid_target_iter):
        yield each_x_valid, each_y_valid