import os
import pickle
import torch
from model import ClassifierModel


class Playground:
    """
    This class is built for testing and playing around with the model.
    """

    def __init__(self, config_params: dict, model: ClassifierModel):
        """
        Method initializes the class to set the configuration and initial variables
        :param config_params: configuration parameters include all relevant data for the process
        :param model: Classifier model which will be used for the task
        """
        self.configuration = self.set_configuration(config_params)
        self.model = model
        self.id2token = self.set_inverse_vocab(is_data=True)
        self.id2label = self.set_inverse_vocab(is_data=False)

    def set_inverse_vocab(self, is_data: bool) -> dict:
        """
        Method is used to generate idx to vocabulary / label dictionary according to vocabulary / label dictionary
        :param is_data: boolean variable specifies whether tokens (True) or labels (False) will be processed
        :return: dictionary which keys are indexes and values are corresponding tokens (True) / labels (False)
        """
        main_source = self.configuration['vocabulary'] if is_data else self.configuration['label_dict']
        return {idx: token for token, idx in main_source.items()}

    @staticmethod
    def set_configuration(parameters: dict) -> dict:
        """
        Method sets main configuration parameters for this class out of all parameters
        :param parameters: configuration parameters include all relevant data for the process
        :return: dictionary which includes following data:
                'checkpoints_dir': path to the checkpoints of each epoch
                'environment': main experimental environment, in which all corresponding train results are kept
                'experiment_num': number of the experiment which is significant to eliminate confusion
                'vocabulary': vocabulary dictionary that is used to encode tokens
                'label_dict': dictionary to encode labels
                'cased': specifies whether model is case-sensitive or not
                'device': device that model was trained on
                'window_shift': step size for shifting the window
                'window_size': number of tokens that window can contain
        """

        checkpoints_dir = os.path.join(parameters['experiment_environment'], 'checkpoints')
        return {
            'checkpoints_dir': checkpoints_dir,
            'environment': parameters['experiment_environment'],
            'experiment_num': parameters['experiment_number'],
            'vocabulary': parameters['vocabulary'],
            'label_dict': parameters['label_dict'],
            'cased': parameters['cased'],
            'device': parameters['device'],
            'window_shift': parameters['window_shift'],
            'window_size': parameters['window_size']
        }

    def get_best_model(self) -> str:
        """
        Method is used to filter epochs according to the F1 score. Path to the best model is returned
        :return: path to the best model parameters
        """
        result_file = os.path.join(self.configuration['environment'],
                                   f"lstm_{self.configuration['experiment_num']}_results.pickle")
        if os.path.exists(result_file):
            with open(result_file, 'rb') as result_data:
                result_dict = pickle.load(result_data)
        specific_dict = {epoch: result['f1_dev'] for epoch, result in result_dict.items()}
        best_epoch = max(specific_dict, key=specific_dict.get)
        best_path = str()
        for ckpt in os.listdir(self.configuration['checkpoints_dir']):
            if f'epoch_{best_epoch}_' in ckpt:
                best_path = ckpt
                break
        return os.path.join(self.configuration['checkpoints_dir'], best_path)

    def load_model(self) -> None:
        """
        Model is used to load state dictionary of the best model which was chosen according to F1 score result
        """
        best_model_path = self.get_best_model()
        self.model.load_state_dict(torch.load(best_model_path))

        self.model.eval()

    def check_remap(self, windows, sentence):
        """

        :param windows:
        :param sentence:
        :return:
        """

        pred_windows = list()
        for idx in range(0, len(sentence), self.configuration['window_shift']):
            pred_windows.append(sentence[idx: idx + self.configuration['window_size']])

        check_slice = slice(0, self.configuration['window_shift'], 1)

        req_preds = list()
        req_originals = list()
        for pred, original in zip(pred_windows, windows):
            req_org = original[check_slice]
            req_pred = pred[check_slice]
            if '<PAD>' in original[check_slice]:
                idx = req_org.index('<PAD>')
                req_pred = req_pred[0: idx]
                req_org = req_org[0: idx]

            req_preds.append(req_pred)
            req_originals.append(req_org)

        predictions = list()
        for each in req_preds:
            predictions.extend(each)
        return predictions

    def decode_sentence(self, sentence: list, windows: list) -> str:
        """
        Method is used to decode sentence according to its 'windowed' version
        :param sentence: list of tokens in the provided sentence
        :param windows: windows of the sentence for encoding
        :return: decoded (in terms of labels) version of the provided sentence
        """
        pred_sentence = self.check_remap(windows, sentence)
        decoded = [self.decode_token(token) for token in pred_sentence]
        return ''.join(decoded)

    def encode_token(self, token: str) -> str:
        """
        Method is used to encode provided token
        :param token: character from the sentence
        :return: encoding of the provided token
        """
        token = token if self.configuration['cased'] else token.lower()
        return self.configuration['vocabulary'][token] if token in self.configuration['vocabulary'].keys() \
            else self.configuration['vocabulary']['<UNK>']

    def encode_sentence(self, sentence: str) -> tuple:
        """
        Method is used to encode provided sentence
        :param sentence: list of tokens of the sentence
        :return: tuple that contains:
                encoded_window: list of windows in which tokens are kept as encoded
                data: list of windows in which raw tokens are kept
        """
        data = list()
        sentence = [each for each in sentence]
        for idx in range(0, len(sentence), self.configuration['window_shift']):
            window = sentence[idx: idx + self.configuration['window_size']]
            if len(window) < self.configuration['window_size']:
                window += ['<PAD>'] * (self.configuration['window_size'] - len(window))
            data.append(window)
        encoded_window = list()
        for each_window in data:
            enc_wind = [self.encode_token(token) for token in each_window]
            encoded_window.append(enc_wind)

        return encoded_window, data

    def decode_token(self, idx: int) -> str:
        """
        Method is used to decode the provided index according to the vocabulary data of the project
        :param idx: string character
        :return: token which corresponds to the index. In case index is higher than length of vocab, it will return
                '<UNK>'
        """
        return self.id2label[idx]

    def run_inference(self, sentence: str) -> None:
        """
        Method is used to perform inference with following order:
            - Load the model
            - encode the sentence
            - predict
            - decode prediction
            - print result
        :param sentence: input sentence which is provided by user
        :return: None
        """
        self.load_model()

        encoded_sentence, data_form = self.encode_sentence(sentence)
        input_data = torch.LongTensor(encoded_sentence).to(self.configuration['device'])
        output = self.model(input_data)
        prediction = torch.argmax(output.view(-1, output.shape[-1]), dim=1).tolist()
        result = self.decode_sentence(prediction, data_form)
        print('Classification result: ')
        print(sentence)
        print(result)

    def process(self):
        """
        Method is used as main function of playground phase. It takes sentence of the user and send it for process
        :return: None
        """
        sentence = input('Please write your sentence:')
        self.run_inference(sentence)
