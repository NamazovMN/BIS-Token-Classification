import collections
import matplotlib.pyplot as plt
import os
import pickle
import seaborn as sns
from collections import Counter
from sklearn.metrics import confusion_matrix


class Statistics:
    """
    Class is used to evaluate all possible statistics before and after the training phase
    """

    def __init__(self, config_parameters: dict):
        """
        Method initializes the class to set the configuration and initial variables
        :param config_parameters: configuration parameters include all relevant data for the process
        """
        self.configuration = self.set_configuration(config_parameters)

    @staticmethod
    def set_configuration(parameters: dict) -> dict:
        """
        Method sets main configuration parameters for this class out of all parameters
        :param parameters: configuration parameters include all relevant data for the process
        :return: dictionary which includes required data for statistics
        """
        result_data = os.path.join(parameters['experiment_environment'],
                                   f"lstm_{parameters['experiment_number']}_results.pickle")

        return {
            'init_stats': parameters['init_statistics'],
            'cased': parameters['cased'],
            'environment': parameters['experiment_environment'],
            'results': result_data,
            'data_path': os.path.join(parameters['dataset_dir'], 'en.wiki.sentences.train'),
            'gold_path': os.path.join(parameters['dataset_dir'], 'en.wiki.gold.train'),
            'vocabulary': parameters['vocabulary'],
            'label_dict': parameters['label_dict']
        }

    def visualize_vocabulary(self) -> None:
        """
        Method is used to visualize several data from vocabulary and label to idx dictionary
        :return: None
        """
        if self.configuration['init_stats']:
            print('The first 5 elements from vocabulary: ')
            for count, (token, idx) in enumerate(self.configuration['vocabulary'].items()):
                if count == 5:
                    break
                print(f'{token}: {idx}')
            for token, idx in self.configuration['label_dict'].items():
                print(f'{token}: {idx}')

    def visualize_stats(self, is_data: bool = True) -> str:
        """
        Method is used to visualize statistics about token / label occurrences in Dataset
        :param is_data: boolean variable specifies whether we analyze vocabulary (it must be limited) or labels
        :return: the label with the highest frequency in train dataset
        """
        source_file = self.configuration['data_path'] if is_data else self.configuration['gold_path']
        set_file = open(source_file)
        all_list = list()
        for each_line in set_file:
            new_line = each_line.replace('\n', '')
            if is_data and not self.configuration['cased']:
                new_line = (new_line.lower())
            all_list.extend(list(new_line))

        count_dict = Counter(all_list)
        num_data = 10 if is_data else len(count_dict)
        if self.configuration['init_stats']:
            print(f"Frequency visualization of {'targets' if is_data else 'labels'} in train dataset")

        for idx, (val, count) in enumerate(count_dict.items()):
            if idx == num_data and is_data:
                break
            if self.configuration['init_stats']:
                print(f'{val} occurs {count} times')
        return max(count_dict, key=count_dict.get)

    @staticmethod
    def get_best_results(result_file: str) -> int:
        """
        Method investigate trained model results and returns the epoch choice according to the f1 score of model on dev
        dataset
        :param result_file: path to the file in which results of model after each epoch exist
        :return: if result file exists (after training) then the last epoch will be returned, otherwise 0
        """
        epoch_num = 0
        if os.path.exists(result_file):
            with open(result_file, 'rb') as result_dict:
                result = pickle.load(result_dict)
            specific_results = dict()
            for epoch, results in result.items():
                specific_results[epoch] = results['f1_dev']
            epoch_num = max(specific_results, key=specific_results.get)
        return epoch_num

    def generate_confusion_matrix(self, inference_data: str) -> None:
        """
        Method is used to compute confusion matrix according to the provided inference data
        :param inference_data: path to the file, in which inference data is kept
        :return: None
        """
        with open(inference_data, 'rb') as pred_file:
            pred_data = pickle.load(pred_file)

        conf_matrix = confusion_matrix(pred_data['targets'], pred_data['predictions'])

        plt.figure(figsize=(8, 6), dpi=100)
        sns.set(font_scale=1.1)

        ax = sns.heatmap(conf_matrix, annot=True, fmt='d', )

        ax.set_xlabel("Predicted Labels", fontsize=14, labelpad=20)
        ax.xaxis.set_ticklabels(['S', 'B', 'I'])

        ax.set_ylabel("Actual Labels", fontsize=14, labelpad=20)
        ax.yaxis.set_ticklabels(['S', 'B', 'I'])

        ax.set_title("Confusion Matrix for BIS", fontsize=14, pad=20)
        image_name = os.path.join(self.configuration['environment'], 'confusion_matrix.png')
        plt.savefig(image_name)
        plt.show()

    def plot_results(self, is_accuracy: bool = True) -> None:
        """
        Method is used to plot accuracy/loss graphs after training session is over, according to provided variable
        :param is_accuracy: boolean variable specifies the type of data will be plotted
        :return: None
        """
        metric_key = 'accuracy' if is_accuracy else 'loss'
        dev_data = list()
        train_data = list()
        with open(self.configuration['results'], 'rb') as result_data:
            result_dict = pickle.load(result_data)
        ordered = collections.OrderedDict(sorted(result_dict.items()))

        for epoch, results in ordered.items():
            dev_data.append(results[f'dev_{metric_key}'])
            train_data.append(results[f'train_{metric_key}'])
        plt.figure()
        plt.title(f'{metric_key.title()} results over {len(result_dict.keys())} epochs')
        plt.plot(list(result_dict.keys()), train_data, 'g', label='Train')
        plt.plot(list(result_dict.keys()), dev_data, 'r', label='Validation')
        plt.xlabel('Number of epochs')
        plt.ylabel(f'{metric_key.title()} results')
        plt.legend(loc=4)
        figure_path = os.path.join(self.configuration['environment'], f'{metric_key}_plot.png')
        plt.savefig(figure_path)
        plt.show()

    def provide_statistics(self, before_training: bool = True) -> str:
        """
        Method is used to provide statistics either before (data statistics) or after the training (results)
        :param before_training: boolean variable to specify whether statistics will be done before or after the training
        :return: the most frequent label which will be useful for baseline F1 score computation
        """
        if before_training:
            self.visualize_vocabulary()
            _ = self.visualize_stats()
            most_freq_label = self.visualize_stats(is_data=False)
            return most_freq_label
        else:
            if os.path.exists(self.configuration['results']):
                chosen_epoch = self.get_best_results(self.configuration['results'])
                inference_data = os.path.join(self.configuration['environment'],
                                              f"inferences/inference_epoch_{chosen_epoch}.pickle")
                self.plot_results(is_accuracy=True)
                self.plot_results(is_accuracy=False)
                self.generate_confusion_matrix(inference_data)
            else:
                raise Exception('There is not any trained model! Please, train the model first!')
            return 'None'
