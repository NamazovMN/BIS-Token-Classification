import os

class Data_Reader(object):
  def __init__(self, path_drive, file_name):
    
    self.generate_file_path(path_drive, file_name)

  # generate the file path in order to read data from the fiven file path
  def generate_file_path(self, path_drive, file_name):
      self.data_path = os.path.join(path_drive, file_name)
      print("The training file: {} has been set".format(file_name))
  # does not hold all data in memory. By using it we can iterate through all data.
  def __iter__(self):
    file = open(self.data_path, 'r', encoding = 'utf-8')
    for each_line in file:
      yield each_line
  # we can get any sentence by calling this method
  def idx_sentence(self, idx):
    file = open(self.data_path, 'r', encoding = 'utf-8')
    i = 0
    for each in file:
      if i == idx:
        return each
      else:
        i += 1