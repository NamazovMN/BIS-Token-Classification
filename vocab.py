import os


class Vocabulary:
    """
    Class is used to set vocabulary object for the task
    """
    def __init__(self, config_parameters: dict):
        """
        Method initializes the class to set the configuration and initial variables
        :param config_parameters: configuration parameters include all relevant data for the process
        """
        self.configuration = self.set_configuration(config_parameters)
        self.vocabulary = self.create_vocabulary()

    @staticmethod
    def set_configuration(parameters: dict) -> dict:
        """
        Method sets main configuration parameters for this class out of all parameters
        :param parameters: configuration parameters include all relevant data for the process
        :return: dictionary which includes following data:
            'dataset_path': path to raw dataset,
            'cased': defines whether tokens will be case-sensitive (True) or not (False)
        """
        vocabulary = parameters['vocabulary'] if 'vocabulary' in parameters.keys() else None
        dataset_path = os.path.join(parameters['dataset_dir'], 'en.wiki.sentences.train')
        return {
            'vocabulary': vocabulary,
            'dataset_path': dataset_path,
            'cased': parameters['cased'],
        }

    def collect_tokens(self) -> list:
        """
        Method is used to collect tokens from the train dataset, since vocabulary is set on the train dataset
        :return: list of unique tokens
        """
        dataset_file = open(self.configuration['dataset_path'])
        tokens = list()
        for each_line in dataset_file:
            clean_line = each_line.replace('\n', '')

            tokens.extend(list(clean_line))

        unique_tokens = [self.set_lower(each_token) for each_token in set(tokens)]
        return [each for each in set(unique_tokens)]

    def set_lower(self, token: str) -> str:
        """
        Method checks whether scenario is case-sensitive or not, and returns tokens, accordingly
        :param token: character token from the dataset
        :return: token itself (case-sensitive) / lowered token (not-case-sensitive)
        """
        if token.isupper and not self.configuration['cased']:
            return token.lower()
        else:
            return token

    def create_vocabulary(self) -> dict:
        """
        Method generates vocabulary object for the task
        :return: dictionary, in which keys are tokens and values are their indexes in the vocabulary
        """

        if self.configuration['vocabulary']:
            vocabulary = self.configuration['vocabulary']
        else:
            token_list = self.collect_tokens()

            vocabulary = {token: idx for idx, token in enumerate(token_list)}
            vocabulary['<PAD>'] = len(vocabulary)
            vocabulary['<UNK>'] = len(vocabulary)
        return vocabulary

    def __iter__(self):
        """
        Method is used to transform vocabulary object to generator, by which we can iterate through te vocabulary
        """
        for each_char, each_idx in self.vocabulary.items():
            yield {each_char: each_idx}

    def __len__(self) -> int:
        """
        Method indicates the number of elements in the vocabulary
        :return: length of vocabulary
        """
        return len(self.vocabulary)

    def __getitem__(self, char: str) -> int:
        """
        Method returns character's index from the vocabulary
        :param char: token that is requested from the vocabulary
        :return: requested element's index
        """
        if char in self.vocabulary.keys():
            return self.vocabulary[char]
        else:
            return self.vocabulary['<UNK>']
