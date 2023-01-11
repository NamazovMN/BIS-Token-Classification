import os
import pickle
import torch
from torch.utils.data import Dataset
from process_data import ReadDatasets
from tqdm import tqdm


class TokenDataset(Dataset):
    """Token dataset class which is inherited from the Dataset object from torch"""

    def __init__(self, config_params: dict, dataset_type: str, vocabulary):
        """
        Method initializes the class to set the configuration and initial variables
        :param config_params: configuration parameters include all relevant data for the process
        :param dataset_type: type of dataset which can either be training dataset or dev dataset
        """
        self.data_type = dataset_type
        self.configuration = self.set_configuration(config_params, dataset_type, vocabulary)
        self.data, self.label = self.get_dataset()

    @staticmethod
    def set_configuration(parameters: dict, dataset_type: str, vocabulary) -> dict:
        """
        Method sets main configuration parameters for this class out of all parameters
        :param vocabulary: Vocabulary object for this classification task
        :param dataset_type: type of dataset which can either be training dataset or dev dataset
        :param parameters: configuration parameters include all relevant data for the process
        :return: dictionary which includes following data:
            'length': number of data in the dataset
            'process_object': object to read and process the raw data
        """
        process_obj = ReadDatasets(parameters, dataset_type, vocabulary)
        ds_file = os.path.join(process_obj.configuration['processed_dir'], f'{dataset_type}.pickle')

        with open(ds_file, 'rb') as inp_data:
            ds = pickle.load(inp_data)
        return {
            'length': len(ds['labels']),
            'process_object': process_obj
        }

    def get_dataset(self) -> tuple:
        """
        Method is used to collect data for required form of Dataset
        :return: tuple that contains data and label tensors
        """
        data = list()
        label = list()
        ti = tqdm(iterable=self.configuration['process_object'], total=self.configuration['length'],
                  desc=f'{self.data_type.title()} Dataset is prepared: ')
        for each_data, each_label in ti:
            data.extend(each_data)
            label.extend(each_label)
        return torch.LongTensor(data), torch.LongTensor(label)

    def __getitem__(self, item: int) -> dict:
        """
        Method is used to retrieve data and label for provided index
        :param item: idx of data in the Dataset
        :return: dictionary, which contains following information:
                'data': tensor of input data,
                'label': tensor of label sequence of corresponding input data
        """
        return {
            'data': self.data[item],
            'label': self.label[item]
        }

    def __len__(self) -> int:
        """
        Method is used to provide information about length of dataset
        :return: integer specifies the length of the Dataset
        """
        return len(self.data)
