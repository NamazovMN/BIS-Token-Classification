import os
import pickle
from vocab import Vocabulary


class ReadDatasets:
    """
    Class is used to read and process raw data from the given datasets
    """

    def __init__(self, config_parameters: dict, dataset_type: str, vocabulary):
        """
        Method initializes the class to set the configuration and initial variables
        :param config_parameters: configuration parameters include all relevant data for the process
        :param dataset_type: type of dataset which can either be training dataset or dev dataset
        """
        self.configuration = self.set_configuration(config_parameters, vocabulary)
        self.create_datasets(dataset_type)
        self.dataset_type = dataset_type

    def set_configuration(self, parameters: dict, vocabulary: Vocabulary) -> dict:
        """
        Method sets main configuration parameters for this class out of all parameters
        :param parameters: configuration parameters include all relevant data for the process
        :param vocabulary:  vocabulary object which was created according to the training dataset
        :return: dictionary which includes following data:
            'dataset_path': path to raw dataset,
            'processed_dir': path to pre-processed data, which will be processed in this class
            'vocabulary': vocabulary dictionary for the model
            'label_dict': dictionary for keeping label to idx information
            'window_shift': step size to shift tokens' windows
            'window_size': size of tokens' windows
        """
        processed_dir = self.check_dir('processed_datasets')
        return {
            'dataset_path': parameters['dataset_dir'],
            'processed_dir': processed_dir,
            'vocabulary_obj': vocabulary,
            'label_dict': parameters['label_dict'],
            'window_shift': parameters['window_shift'],
            'window_size': parameters['window_size']
        }

    @staticmethod
    def check_dir(directory: str) -> str:
        """
        Method checks whether the provided path exist or not. In case it does not exist, path will be created
        :param directory: path, which existence will be checked
        :return: path, which existence was assured
        """
        if not os.path.exists(directory):
            os.makedirs(directory)
        return directory

    def create_datasets(self, dataset_type: str) -> None:
        """
        Method combines data and their ground truth from different files in one file
        :param dataset_type: type of data that is collected. It can either be training dataset or dev dataset
        :return: None
        """

        out_dir = os.path.join(self.configuration['processed_dir'], f'{dataset_type}.pickle')
        if not os.path.exists(out_dir):
            sentences_file = open(os.path.join(self.configuration['dataset_path'],
                                               f'en.wiki.sentences.{dataset_type}'), 'r')
            labels_file = open(os.path.join(self.configuration['dataset_path'],
                                            f'en.wiki.gold.{dataset_type}'), 'r')
            sentences = list()
            labels = list()
            for each_sentence, each_label in zip(sentences_file, labels_file):
                sentences.append([each_token for each_token in each_sentence.replace('\n', '')])
                labels.append([label for label in each_label.replace('\n', '')])
            result = {
                'sentences': sentences,
                'labels': labels
            }
            with open(out_dir, 'wb') as out_file:
                pickle.dump(result, out_file)

    def check_token(self, token, is_label):
        """
        Method check whether token exists in vocabulary and indexes it accordingly. If label is provided, checking
        process is skipped
        :param token: can either be character or label
        :param is_label: boolean variable that allows to check existence for labels
        :return: index of token/label in vocabulary/label to idx dictionary
        """
        if is_label:
            return self.configuration['label_dict'][token]
        else:
            return self.configuration['vocabulary_obj'][token]

    def encode_sentence(self, sentence: list, is_label: bool) -> list:
        """
        Method encodes provided sentence according to information that sentence is sequence of tokens or labels
        :param sentence: list of tokens/labels in provided sentence
        :param is_label: boolean variable specifies whether provided list is sequence of labels or tokens
        :return: list of encoded tokens/labels according to the provided is_label information
        """
        return [self.check_token(token, is_label) for token in sentence]

    def make_windows(self, sentence: list, is_label: bool) -> list:
        """
        Method creates windows in provided sentence according to window size and window shift information
        :param sentence: Any sentence from the dataset (in form of list)
        :param is_label: boolean variable defines whether we encode labels or tokens
        :return: list of windows (which is also list) of tokens (int)
        """
        sentence_list = list()
        for idx in range(0, len(sentence), self.configuration['window_shift']):
            window = sentence[idx: idx + self.configuration['window_size']]
            if len(window) < self.configuration['window_size']:
                window += ['<PAD>'] * (self.configuration['window_size'] - len(window))
            sentence_list.append(self.encode_sentence(window, is_label))
        return sentence_list

    def __iter__(self):
        """
        Method is used to make the class perform as generator, which run through the data and processes it.
        """
        dataset_file = os.path.join(self.configuration['processed_dir'], f'{self.dataset_type}.pickle')
        with open(dataset_file, 'rb') as ds_data:
            dataset = pickle.load(ds_data)
        for sentence, label in zip(dataset['sentences'], dataset['labels']):
            yield self.make_windows(sentence, is_label=False), self.make_windows(label, is_label=True)
