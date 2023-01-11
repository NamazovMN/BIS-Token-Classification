import pickle
from dataset import TokenDataset
from model import ClassifierModel
from playground import Playground
from statistics import Statistics
from torch.utils.data import DataLoader
from trainer import Train
from utilities import *
from vocab import Vocabulary


def set_configuration() -> dict:
    """
    Function is used to collect parameters from the user and generate configuration parameters for the project
    :return: dictionary that includes initial configuration parameters
    """
    arguments = create_parser()
    parameters = collect_parameters(arguments)
    return parameters


def get_datasets(parameters: dict, vocab_object: Vocabulary) -> dict:
    """
    Function is used to collect datasets and generate DataLoaders for the model
    :param parameters: configuration parameters for the project
    :param vocab_object: vocabulary object that is needed to encode data
    :return: dictionary contains training and validation DataLoaders
    """
    train_ds = TokenDataset(parameters, 'train', vocab_object)
    dev_ds = TokenDataset(parameters, 'dev', vocab_object)

    return {
        'train': DataLoader(train_ds, parameters['batch_size'], shuffle=True),
        'dev': DataLoader(dev_ds, parameters['batch_size'], shuffle=True)
    }


def get_vocabularies(parameters: dict) -> tuple:
    """
    Function collects vocabulary, label dictionary and maximum length information according to the provided datasets. If
    condition is activated when user wants to continue training, so that vocabulary must be same as before.
    :param parameters: configuration parameters for the project
    :return: tuple object contains:
            vocabulary: vocabulary object which is used to encode dataset
            label_dict: dictionary contains labels and their indexes
            max_length: length of the longest sequence in train and validation datasets
    """
    experimental_config = os.path.join(
        f'train_results/experiment_{parameters["experiment_number"]}/model_config.pickle')
    if os.path.exists(experimental_config):
        with open(experimental_config, 'rb') as param_dict:
            old_params = pickle.load(param_dict)
        vocabulary = Vocabulary(old_params)
        label_dict = old_params['label_dict']
        max_length = old_params['max_length']
    else:
        vocabulary = Vocabulary(parameters)
        label_dict, max_length = get_required(parameters['dataset_dir'])
    return vocabulary, label_dict, max_length


def __main__():
    config_parameters = set_configuration()
    vocabulary, label_dict, max_length = get_vocabularies(config_parameters)

    config_parameters['max_length'] = max_length
    config_parameters['label_dict'] = label_dict
    config_parameters['vocabulary'] = vocabulary.vocabulary

    datasets = get_datasets(config_parameters, vocabulary)
    model = ClassifierModel(config_parameters).to(config_parameters['device'])

    trainer = Train(config_parameters, model)
    config_parameters['experiment_environment'] = trainer.configuration['environment']

    statistics = Statistics(config_parameters)
    max_label = statistics.provide_statistics(before_training=True)

    if config_parameters['train_model']:
        trainer.train_epoch(datasets, max_label)
    statistics.provide_statistics(before_training=False)
    player = Playground(config_parameters, model)
    player.process()


if __name__ == '__main__':
    __main__()
