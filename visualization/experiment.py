import bittensor
import torch

import sys
import argparse
import time
import bittensor
import torch
import os
import wandb
import math
import pandas
import traceback
from rich import print
from rich.console import Console
from rich.traceback import install
import torch.nn as nn

from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from loguru import logger
import pandas as pd

logger = logger.opt( colors=True )
console = Console()
install(show_locals=True)


class neuron:
    """ Only for getting other's representation 
    """
    def __init__( self, config: 'bittensor.Config' = None ):

        # === Set up Config ===
        if config == None: config = neuron.config()
        self.config = config
        neuron.check_config( self.config )
        self.config.to_defaults()
        if self.config.neuron._mock == True:
            self.config.subtensor._mock = True
            self.config.wallet._mock = True
            self.config.dataset._mock = True
            self.config.dendrite._mock = True
            self.config.metagraph._mock = True
            self.config.subtensor._mock = True
        print ( self.config )

        # === Create Bittensor objects ===
        bittensor.logging( config = self.config, logging_dir = self.config.neuron.full_path )
        self.wallet = bittensor.wallet ( config = self.config )
        self.subtensor = bittensor.subtensor ( config = self.config )
        self.metagraph = bittensor.metagraph ( config = config, subtensor = self.subtensor )        
        self.dendrite = bittensor.dendrite ( config = self.config) # , wallet = self.wallet )
        self.device = torch.device ( device = self.config.neuron.device )    
        self.dataset = bittensor.dataset ( config = self.config, batch_size = self.subtensor.validator_batch_size, block_size = self.subtensor.validator_sequence_length )
        
        
        self.metagraph.sync()

    def __del__(self):
        self.dataset.close()

    @classmethod
    def check_config( cls, config: 'bittensor.Config' ):
        r""" Checks/validates the config namespace object.
        """
        bittensor.logging.check_config( config )
        bittensor.wallet.check_config( config )
        bittensor.subtensor.check_config( config )
        bittensor.metagraph.check_config( config )
        bittensor.dataset.check_config( config )
        bittensor.dendrite.check_config( config )
        bittensor.wandb.check_config( config )
        full_path = os.path.expanduser('{}/{}/{}/{}'.format( config.logging.logging_dir, config.wallet.name, config.wallet.hotkey, config.neuron.name ))
        config.neuron.full_path = os.path.expanduser(full_path)
        config.using_wandb = config.wandb.api_key != 'default'
        if not os.path.exists(config.neuron.full_path):
            os.makedirs(config.neuron.full_path)

    @classmethod
    def add_args( cls, parser ):
        parser.add_argument('--neuron.name', type=str, help='Trials for this miner go in miner.root / (wallet_cold - wallet_hot) / miner.name ', default='core_validator')
        parser.add_argument('--neuron.blocks_per_epoch', type=int, help='Blocks per epoch, -1 value means we use the chain value.', default = -1 )
        parser.add_argument('--neuron.epochs_until_reset', type=int, help='Number of epochs before weights are reset.', default = -1 )
        parser.add_argument('--neuron.device', type=str, help='miner default training device cpu/cuda', default=("cuda" if torch.cuda.is_available() else "cpu"))
        parser.add_argument('--neuron.restart_on_failure',  action='store_true', help='''Restart neuron on unknown error.''', default=True )
        parser.add_argument('--neuron._mock', action='store_true', help='To turn on neuron mocking for testing purposes.', default=False )

    @classmethod
    def config ( cls ):
        parser = argparse.ArgumentParser()    
        cls.add_args( parser )
        bittensor.wallet.add_args( parser )
        bittensor.dendrite.add_args( parser )
        bittensor.subtensor.add_args( parser )
        bittensor.metagraph.add_args( parser )
        bittensor.logging.add_args( parser )
        bittensor.dataset.add_args( parser )
        bittensor.wandb.add_args(parser)
        return bittensor.config( parser )


    def forward(self):
        return self.dendrite.forward_text (
            endpoints = self.metagraph.endpoints,
            inputs = next(self.dataset)
        )

class experiment:
    def __init__(self):
        config = neuron.config()
        config.wallet.name = 'bittensor'
        config.wallet.hotkey = 'vis'
        config.neuron.name = 'client'
        self.config = config
        self.client = neuron(config)
        self.results = pd.DataFrame(columns = ['uid', 'batch_size', 'sequence_len', 'code', 'time'])

    def run(self):
        print('exp running')

        for sequence_len in range(10, 550, 100):
            self.client.dataset.set_data_size(batch_size = 10, block_size = sequence_len)
            representations, codes, times = self.client.forward() 
            print('client.forward finished')
            result = pd.DataFrame({'uid': range(2000), 'code': codes.detach(), 'time': times.detach()})
            result['batch_size'] = self.config.dataset.batch_size
            result['sequence_len'] = sequence_len
            self.results = pd.concat([self.results, result])
        
        
        self.results.to_csv('/home/isabella/.bittensor/bittensor/visualization/results.csv', index = False)
        return self.results


if __name__ == "__main__":
    experiment().run()