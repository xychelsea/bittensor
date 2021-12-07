""" Implementation for the dataset and GenesisTextDataset class, which handles dataloading from ipfs
"""
# The MIT License (MIT)
# Copyright © 2021 Yuma Rao

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import os
import random
from re import I

from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Subset
import torch

from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import requests
import re 
import nltk
import json
import yaml

nltk.download('wordnet')


from loguru import logger
import bittensor

logger = logger.opt(colors=True)


class Dataset():
    """ Implementation for the dataset class, which handles dataloading from ipfs
    """
    def __init__(self):
        
        # Used to retrieve directory contentx
        self.cat = 'http://ipfs.opentensor.ai/api/v0/cat' 
        self.node_get = 'http://ipfs.opentensor.ai/api/v0/object/get'
        self.mountain_hash = 'QmSdDg6V9dgpdAFtActs75Qfc36qJtm9y8a7yrQ1rHm7ZX'
        # Used when current corpus has been exhausted
        self.refresh_corpus = False
        

    @staticmethod
    def requests_retry_session(
            retries=10,
            backoff_factor=0.5,
            status_forcelist=(104, 500, 502, 504),
            session=None,
        ):
        """ Creates a retriable session for request calls. This enables
        automatic retries and back-off retries should any request calls fail.

        Args:
            retries (int, optional): Maximum number of retries. Defaults to 3.
            backoff_factor (float, optional): Factor by which to back off if a retry fails. Defaults to 0.3.
            status_forcelist (tuple, optional): A set of integer HTTP status codes that we should force a retry on. Defaults to (500, 502, 504).
            session ([type], optional): Session for which to set up the retries. Defaults to None.

        Returns:
            requests.Session(): A Requests Session object set up for retries and backoff.
        """

        session = session or requests.Session()
        retry = Retry(
            total=retries,
            read=retries,
            connect=retries,
            backoff_factor=backoff_factor,
            status_forcelist=status_forcelist,
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount('http://', adapter)
        session.mount('https://', adapter)
        return session

    def retrieve_directory(self, address: str, params = None, action: str = 'post'):
        r"""Connects to Pinata IPFS gateway and retrieves directory.

        Returns:
            dict: A dictionary of the files inside of the genesis_datasets and their hashes.
        """
        session = requests.Session()
        session.params.update(params)
        if action == 'get':
            response = Dataset.requests_retry_session(session=session).get(address)
        elif action == 'post':
            response = Dataset.requests_retry_session(session=session).post(address)
        return response

    def __len__(self):
        """ Returns length of the dataset that the dataset is processing
        """

    def __getitem__(self, idx):
        """ Returns the next batch from the dataset.
        """
class DataPreprocessing():
    def __init__(
        self,
        use_default_preprocesses = False,
    ):
        self.use_default_preprocesses = use_default_preprocesses
        
        preprocessing_config_path = os.path.expanduser("~/.bittensor/bittensor/bittensor/_dataset/data_preprocessing.yaml")
        with open(preprocessing_config_path, "r") as stream:
            self.preprocesses_per_dataset = yaml.safe_load(stream)

        self.default_processes = {
            'BookCorpus2': [self.standardise_quotations, self.remove_next_line, self.remove_multiple_spaces],
            'Books3': [self.standardise_quotations, self.remove_next_line, self.remove_multiple_spaces],
            'Gutenberg_PG': [self.standardise_quotations, self.remove_next_line, self.remove_repetitive_character],
            'ArXiv': [self.standardise_quotations, self.remove_next_line, self.remove_latext_math, self.remove_cite, self.remove_tags, self.remove_repetitive_character, self.remove_multiple_spaces],
            'PhilPapers': [self.standardise_quotations, self.remove_next_line, self.remove_latext_math, self.remove_cite, self.remove_tags, self.remove_repetitive_character, self.remove_multiple_spaces],
            'HackerNews': [self.standardise_quotations, self.remove_next_line, self.remove_http_links, self.remove_cite, self.remove_repetitive_character, self.remove_multiple_spaces],
            'OpenSubtitles': [self.standardise_quotations, self.remove_next_line, self.remove_double_quote, self.remove_multiple_spaces],
            'YoutubeSubtitles': [self.standardise_quotations, self.remove_next_line, self.remove_multiple_spaces, self.lower_case],
            'UbuntuIRC': [self.standardise_quotations, self.remove_next_line, self.remove_multiple_spaces, self.remove_http_tags],
            'default': [self.standardise_quotations, self.remove_next_line, self.remove_multiple_spaces]
        }

        self.stemmer = nltk.stem.snowball.SnowballStemmer("english")
        self.lemmatizer = nltk.stem.WordNetLemmatizer()
        self.stop_words = set(nltk.corpus.stopwords.words('english'))
        
        self.word_by_word_processes_seq = [
            self.remove_stopwords,
            self.stemming,
            self.lemmatization
        ]

        self.joint_words_processes_seq = [
            self.lower_case,
            self.remove_punctuations,
            self.remove_http_links,
            self.remove_tags,
            self.remove_http_tags,
            self.standardise_quotations,
            self.remove_latext_math,
            self.remove_cite, 
            self.remove_next_line,
            self.remove_repetitive_character,
            self.remove_double_quote,
            self.remove_multiple_spaces,
        ]


    def clean(self, directory_key, text):
        if self.use_default_preprocesses:
            if directory_key not in self.default_processes.keys():
                key = 'default'
            else:
                key = directory_key

            default_processes_name = [p.__name__ for p in self.default_processes[key]]
            word_by_word_processes = [ p for p in self.word_by_word_processes_seq if p.__name__ in default_processes_name ]
            joint_words_processes = [ p for p in self.joint_words_processes_seq if p.__name__ in default_processes_name ]

        elif directory_key in self.preprocesses_per_dataset.keys():
            if directory_key not in self.preprocesses_per_dataset.keys():
                key = 'default'
            else:
                key = directory_key

            word_by_word_processes = [ p for p in self.word_by_word_processes_seq if p.__name__ in self.preprocesses_per_dataset[key] ]
            joint_words_processes = [ p for p in self.joint_words_processes_seq if p.__name__ in self.preprocesses_per_dataset[key] ]

        else:
            word_by_word_processes = []
            joint_words_processes = []

        for fun in joint_words_processes:
            text = fun(text)
        
        text = text.split(" ")
        for fun in word_by_word_processes:
            text = fun(text)

        return text

    def remove_nested_parentheses(self, text, open_parent, close_parent):
        n = len(open_parent)
        result = ''
        skip = 0
        skip_n = 0
        for i, t in enumerate(text):
            if text[i:i+n] == open_parent:
                skip += 1
            elif skip_n > 0 :
                skip_n -= 1
            elif text[i:i+n] == close_parent and skip > 0:
                skip -= 1
                skip_n = n-1
            elif skip == 0:
                result += t
        return result

    def remove_stopwords(self, text):
        return [t for t in text if not t.lower() in self.stop_words]
    
    def stemming(self, text):
        return [self.stemmer.stem(t) for t in text]
    
    def lemmatization(self, text):
        return [self.lemmatizer.lemmatize(t) for t in text]
    
    def lower_case(self, text):
        return text.lower()

    def remove_next_line(self, text):
        text = text.replace("\\n", " ")
        text = text.replace("\n", " ")
        text = text.replace("\ ", " ")
        text = self.remove_multiple_spaces(text)
        return text

    def standardise_quotations(self, text):
        text = text.replace("“", '"')
        text = text.replace("”", '"')
        text = text.replace("''", '"')
        text = text.replace("``", '"')
        text = text.replace("\“", '"')
        text = text.replace("\”", '"')
        text = text.replace('\\"', '"')
        return text

    def remove_punctuations(self, text):
        text = self.remove_next_line(self, text)
        regex = r"[^\w\s]"
        text = re.sub(regex, "", text)
        return text

    def remove_latext_math(self, text):
        # remove $ * $ pattern
        regex = r"(\$+)(?:(?!\1)[\s\S])*\1"
        text = re.sub(regex, "", text)
        return text

    def remove_cite(self, text):
        # remove \\[ * \\] pattern
        text = self.remove_nested_parentheses(text, '\\[', '\\]')
        text = self.remove_nested_parentheses(text, '\[', '\]')
        text = self.remove_nested_parentheses(text, '[', ']')
        text = self.remove_nested_parentheses(text, '{', '}')

        return text

    def remove_tags(self, text):
        regex = r'(\\)\w+'
        text = re.sub(regex, "", text)
        return text

    def remove_http_links(self, text):
        text = re.sub(r'http\S+', '', text)
        return text

    def remove_repetitive_character(self, text):
        regex = r'([\*\=\-\~]{2,10})'
        text = re.sub(regex, "", text)
        return text

    def remove_double_quote(self, text):
        text = re.sub(r'\"', "", text)
        return text

    def remove_http_tags(self, text):
        regex = r'\<\w+\>'
        text = re.sub(regex, "", text)
        return text

    def remove_multiple_spaces(self, text):
        text = re.sub(r'([\s]{2,10})', " ", text)
        return text

class GenesisTextDataset( Dataset ):
    """ One kind of dataset that caters for the data from ipfs 
    """
    def __init__(
        self,
        block_size,
        batch_size,
        max_corpus_size,
        num_workers,
        dataset_name,
        data_dir,
        save_dataset,
        max_datasets,
        no_tokenizer,
        use_default_preprocesses,
    ):
        super().__init__()
        self.block_size = block_size
        self.batch_size = batch_size
        self.max_corpus_size = max_corpus_size
        self.num_workers = num_workers
        self.tokenizer = bittensor.tokenizer( version = bittensor.__version__ )
        self.dataset_name = dataset_name
        self.data_dir = data_dir
        self.save_dataset = save_dataset
        self.datafile_size_bound = 262158
        self.max_datasets = max_datasets
        self.__infinite_dataset_iterator = None
        self.no_tokenizer = no_tokenizer

        # Retrieve a random slice of the genesis dataset
        self.data = []
        self.data_remained = []

        # Used to refresh corpus if we've exhausted the whole dataset
        self.refresh_corpus = True

        self.build_hash_table()

        if not os.path.isdir(os.path.expanduser(data_dir)):
            os.makedirs(os.path.expanduser(data_dir))
            
        self.data_preprocessing = DataPreprocessing(
            use_default_preprocesses = use_default_preprocesses,
        )

    def get_random_directories(self):
        r""" Getting directories from a random dataset_hash
        Where a directory could be leading to a data file or a directory file 
        """
        
        # --- Getting directories from a random dataset hash.
        # --- directories: List[ Map{Name: str, Hash: str, Size: int} ]
        i = 0
        directories = [] 
        dataset_hashes_order = list(range(len(self.dataset_hashes)))
        random.shuffle(dataset_hashes_order)
        
        while i < self.max_datasets:
            
            dataset_key = list(self.dataset_hashes.keys())[dataset_hashes_order[i]]
            dataset_hash = self.dataset_hashes[dataset_key]
            i += 1
            logger.success("Loading dataset:".ljust(20) + "<blue>{}</blue>".format(dataset_key))
            response = self.retrieve_directory(self.cat, (('arg', dataset_hash),))
            
            if response.status_code != 200:
                logger.warning("Failed to retrieve directory, ignoring directory:".ljust(20) + "<blue>{}</blue>".format(dataset_key))
            
            else:
                # --- Get the directory links if there is valid response, else check on another dataset_hash 
                directories += [{'Dataset': dataset_key, **r } for r in response.json()]
                logger.success("Loaded dataset:".ljust(20) + "<blue>{}</blue>".format(dataset_key))
                
        if len(directories) == 0:
            directories = None
        
        return directories

    def get_directories(self, keys):
        directories = []
        for key in keys:
            
            if key in self.dataset_hashes.keys():
                logger.success("Loading dataset:".ljust(20) + "<blue>{}</blue>".format(key))
                dataset_hash = self.dataset_hashes[key] 
                response = self.retrieve_directory(self.cat, (('arg', dataset_hash),))
                if response.status_code != 200:
                    logger.warning("Failed to retrieve directory, ignoring directory:".ljust(20) + "<blue>{}</blue>".format(key))
                
                else:
                    # --- Get the directory links if there is valid response, else check on another dataset_hash
                    directories += [{'Dataset': key, **r } for r in response.json()]
                    logger.success("Loaded dataset:".ljust(20) + "<blue>{}</blue>".format(key))
            else:
                logger.error('Incorrect dataset name:'.ljust(20) + " <red>{}</red>.".format(key)+' Must be one of the following {}'.format(bittensor.__datasets__))

        return directories


    def extract_datafile_dir(self, directory):
        r"""
        With recursion, from the given directory, get a directory that leads to a datafile.

        Args:
            directory: Map{ Name: str, Hash: str, Size: int }: 
                The original directory to look up a datafile for.

        Returns:
            directory: Map{ Name: str, Hash: str, Size: int }: 
                A random directory that lead to a datafile.
        """
        # --- If the size of directory is small, it is leads to data file, return the data file.
        if directory['Size'] <= self.datafile_size_bound:
            return directory

        # --- Else, the directory leads to more directories, return a random data file within the directories.
        else:
            response = self.retrieve_directory(self.node_get, (('arg', directory['Hash']),))
            
            # --- Return none if the request failed.
            if response.status_code != 200:
                logger.warning("Failed to retrieve directory, ignoring directory:".ljust(20) + "<blue>{}</blue>".format(directory))
                return None
            
            # --- Pick a random sub_directory, run recursion until we have found a data file
            else:
                sub_directories = response.json()
                if sub_directories and 'Links' in sub_directories.keys() and len(sub_directories['Links']) >= 1:
                    random_sub_directory = random.choice(sub_directories['Links'])

                    # --- Fill the name of the random_sub_directory if it is empty. 
                    if random_sub_directory['Name'] == '':
                        random_sub_directory['Name'] = directory['Name']
                    
                    return self.extract_datafile_dir(random_sub_directory)
                else:
                    logger.warning("Directory seems empty, ignoring directory:".ljust(20) + "<blue>{}</blue>". format(directory['Hash']))
        return None

    def get_text(self, file):
        r"""
        Load the text data from disk if it is already in the the data_dir,
        else download it from IPFS and save it

        Args:
            file: Map{ Name: str, Hash: str, Size: int }
                The directory to get text file from.
        Returns:
            text: str: 
                The text data.
        """
        text = None
        file_name = file['Name']
        file_hash = file['Hash']
        full_path = os.path.expanduser(os.path.join(self.data_dir, file_name))

        # --- Load text from path
        if os.path.exists(full_path):
            try:
                with open(full_path, mode='r') as f:
                    text = f.read()
                logger.success("Loaded:".ljust(20) + "<blue>{}</blue>".format(file_name))
            except Exception:
                logger.warning("Load failed:".ljust(20) + "<blue>{}</blue>".format(file_name))

        # --- If couldnt load from path, download text.
        if text == None:
            response = self.retrieve_directory(self.node_get, (('arg', file_hash),))

            if response.status_code != 200:
                logger.warning("Failed to retrieve file, ignoring file:".ljust(20) + "<blue>{}</blue>".format(file_name))
            else:
                try:
                    text = json.loads(response.text)["data"]
                except Exception:
                    try:
                        text = json.loads(response.text)["Data"]
                    except Exception:
                        text = response.text

                logger.success("Downloaded:".ljust(20) + "<blue>{}</blue>".format(file_name))
                
                # --- Save text if the save_dataset flag is on.
                if self.save_dataset:
                    try:
                        with open(full_path, mode = 'w+') as f:
                            f.write(text)
                            logger.success("Saved:".ljust(20) + "<blue>{}</blue>".format(file_name))
                    except Exception:
                        logger.warning("Save failed:".ljust(20) + "<blue>{}</blue>".format(file_name))

        return text

    def construct_text_corpus(self, min_data_len = 0):
        """ Main function for generating the text data.
        1. Get directories from a random dataset_hash (dataset_hash is the result from calling pin/ls).
        2. Pick a random directory and get the directory that would lead to a datafile.    
        3. Get text from the directory.
        4. Repeat 2,3 until we have reached the max_corpus_size

        Returns:
            text: str: 
                Contents of the text data.
        """
        try:
            logger.success("Retrieving a dataset files from the IPFS gateway...")

            # --- Get directories from a random dataset_hash
            if len(self.dataset_name) == 0:
                directories = self.get_random_directories()
            else:
                directories = self.get_directories(self.dataset_name)
            data_corpus = []

            # --- Generate a random order of the directories
            directory_order = list(range(len(directories)))
            random.shuffle(directory_order)

            # --- Pick random directories and get their text contents.
            if directories:
                total_dataset_size = 0
                total_dataset_len = 0
                i = 0

                # --- Dont stop until the corpus size and the minimum data_length was reached.
                while (total_dataset_size <= self.max_corpus_size) or (total_dataset_len < min_data_len):
                    # --- Get a directory that leads to a datafile.
                    directory = directories[directory_order[i]]


                    random_datafile_dir = self.extract_datafile_dir(directory)
                    
                    if random_datafile_dir == None:
                        pass

                    # --- Get text from the datafile directory
                    try:
                        text = self.get_text(random_datafile_dir)
                        text_list = self.data_preprocessing.clean(directory['Dataset'], text)

                    except Exception as e: 
                        text = None
                        text_list = None

                    if text != None and text_list != None:
                        data_corpus.extend(text_list)
                        total_dataset_size += int(random_datafile_dir['Size'])
                        total_dataset_len += len(text_list)
                    i += 1

                return data_corpus

            logger.error("It appears the directory is empty... Restart your miner to try again.")
            return None

        except Exception as e:
            logger.error("Ran into exception when trying to retrieve dataset from IPFS: {}".format(e))

        return None

    def dataloader(self, epoch_length = 100):
        """ Creates a torch dataloader out of a subclass of this class.

        Args:
            epoch_length (int, optional): The epoch length of the miner. If this length is not set or if it is larger than the dataset,
            then a dataloader for the entire dataset is returned. Otherwise, a dataloader for a subset of the dataset of epoch_length
            is returned. Defaults to None.

        Returns:
            torch.utils.data.dataloader.DataLoader: Pytorch dataloader.
        """
        data_size = epoch_length * self.batch_size * self.block_size
        
        # Make sure the data remained is at least as big as data_size 
        while len(self.data_remained) <= (data_size):
            self.data_remained +=  self.construct_text_corpus(min_data_len = data_size)

        self.data = self.data_remained[:data_size]
        del self.data_remained[:data_size]

        # Datalaoder calls self._getitem_ functions until the self.data uses up, and group the result by batch size
        return DataLoader(self,
                    shuffle=True,
                    batch_size=self.batch_size,
                    num_workers=self.num_workers,
                    drop_last=True,
                    )

    def __next__(self):
        """Returns the next element from the dataset. 
        """
        if self.__infinite_dataset_iterator == None:
            self.__infinite_dataset_iterator = iter([input for input in self.dataloader(1000)]) # should set it to 1000

        try:
            return next(self.__infinite_dataset_iterator)
        
        except StopIteration:
            self.__infinite_dataset_iterator = iter([input for input in self.dataloader(1000)])
            return next(self.__infinite_dataset_iterator)

    def __len__(self):
        """Returns number of samples (blocks) of dataset

        Returns:
            length: int
        """
        if (self.data == None) or (self.block_size == None) or (self.block_size == 0):
            return 0
        return round( len(self.data) / self.block_size )

    def __getitem__(self, idx):
        """ Returns a block of sentences from text dataset.

            Args:
                idx: index of data input

            Returns:
                torch.tensor(dix)
        """
        start_idx = (idx * self.block_size) % len(self.data)
        end_idx = start_idx + self.block_size

        if self.no_tokenizer == False:
            tokenized_text = torch.tensor(self.tokenizer(" ".join(self.data[start_idx:end_idx]), padding=True, truncation=True)['input_ids'], dtype=torch.long)
            
        elif self.no_tokenizer == True:
            tokenized_text = " ".join(self.data[start_idx:end_idx])

        block = tokenized_text[:self.block_size]
        if len(block) < self.block_size:
            return torch.cat((block, torch.tensor([0]*(self.block_size - len(block)))))
        else:
            return block
    
    def build_hash_table(self):
        self.dataset_hashes = {}
        response = self.retrieve_directory(self.node_get, (('arg', self.mountain_hash),))
        for i in response.json()['Links']:
            self.dataset_hashes[i['Name'][:-4]]= i['Hash'] 

