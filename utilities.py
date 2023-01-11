import argparse
import os
import torch.cuda
from argparse import Namespace


def get_required(dataset_dir) -> tuple:
    """
    Function is used to collect required data from datasets. It is defined separately to eliminate possible conflicts
    with other processes
    :param dataset_dir: path to the raw dataset
    :return: tuple which contains:
            label_dict: dictionary contains labels and their indexes
            max(length_list): length of maximum sequence in train and validation datasets
    """
    length_list = list()
    labels_list = list()
    file_name = os.path.join(dataset_dir, 'en.wiki.gold.train')
    dataset_file = open(file_name)
    for each_line in dataset_file:
        data = each_line.replace('\n', '')
        length_list.append(len(data))
        labels_list.extend(list(data))

    label_dict = {label: idx for idx, label in enumerate(set(labels_list))}
    label_dict['<PAD>'] = len(label_dict)

    return label_dict, max(length_list)


def create_parser() -> Namespace:
    """
    Function is used to collect parameters that are modified by user
    :return: Namespace data which contains required parameters for the configuration
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--resume_training', required=False, action='store_true', default=False,
                        help='Activates training from the last epoch was trained if model was trained before')
    parser.add_argument('--batch_size', required=False, type=int, default=32,
                        help='Specifies batch size of the data for training and development')
    parser.add_argument('--epochs', required=False, type=int, default=30,
                        help='Specifies number of epochs to train the model')
    parser.add_argument('--learning_rate', required=False, type=float, default=0.0003,
                        help='Sets learning rate of the training')
    parser.add_argument('--dataset_dir', required=False, type=str, default='PATH_TO_YOUR_DATASET',
                        help='Specifies the path where raw dataset is kept')
    parser.add_argument('--cased', required=False, action='store_true', default=False,
                        help='Specifies whether all tokens will be case sensitive (True) or not (False)')
    parser.add_argument('--init_eval', required=False, action='store_true', default=False,
                        help='Specifies whether model should be evaluated before training starts (True) or not (False)')
    parser.add_argument('--experiment_number', required=False, type=int, default=13,
                        help='Specifies experiment number that user want to execute')
    parser.add_argument('--embedding_dimension', required=False, type=int, default=300,
                        help='Specifies embedding dimension of the LSTM model')
    parser.add_argument('--hidden_dimension', required=False, type=int, default=100,
                        help='Specifies hidden dimension of LSTM layer')
    parser.add_argument('--dropout', required=False, type=float, default=0.3,
                        help='Specifies dropout rate during the training')
    parser.add_argument('--bidirectional', required=False, action='store_true', default=False,
                        help='Specifies Bi-LSTM (True) or LSTM (False) will be used')
    parser.add_argument('--window_size', required=False, type=int, default=100,
                        help='Specifies size of window of the data will be included')
    parser.add_argument('--window_shift', required=False, type=int, default=100,
                        help='Specifies step that window will move on dataset')
    parser.add_argument('--init_statistics', required=False, action='store_true', default=False,
                        help='Specifies whether initial information about data statistics should be provided or not')
    parser.add_argument('--train_model', required=False, action='store_true', default=False,
                        help='Specifies whether model will be trained (True) or not (False)')
    parser.add_argument('--playground', required=False, action='store_true', default=False,
                        help='Specifies whether user wants to use the model to play around the data')
    parser.add_argument('--num_lstm_layers', required=False, type=int, default=1,
                        help='Specifies number of LSTM layers in the base model')
    parser.add_argument('--optimizer', required=False, type=str, default='SGD', choices=['SGD', 'Adam'],
                        help='Specifies optimizer choice of the user that can either be SGD or Adam')
    parser.add_argument('--show_example', required=False, action='store_true', default=False,
                        help='Specifies whether example will be shown after each 10 epochs of training or not')
    parser.add_argument('--playground_only', required=False, action='store_true', default=False,
                        help='Activates only playground mode and bypasess other activities')
    return parser.parse_args()


def collect_parameters(arguments: Namespace) -> dict:
    """
    Function is used to transform Namespace data into dictionary
    :param arguments: provided arguments by the user
    :return: dictionary that contains initial configuration parameters for the project
    """
    parameters = dict()
    for arg in vars(arguments):
        parameters[arg] = getattr(arguments, arg)

    parameters['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    parameters['lin_out'] = 64
    return parameters
