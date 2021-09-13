""" Hold the neuron class which contains all the sub-modules of bittensor. Also init global neuron. 
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
import argparse
from typing import Callable
import wandb

# Bittensor code and protocol version.
__version__ = '1.2.0'
__version_as_int__ = (100 * 1) + (10 * 2) + (1 * 0)  # Integer representation

# Vocabulary dimension.
#__vocab_size__ = len( tokenizer ) + len( tokenizer.additional_special_tokens) + 100 # Plus 100 for eventual token size increase.
__vocab_size__ = 50378

# Tensor dimension.
# NOTE (const): if/when this increases peers must be responsible for trimming or expanding output to this size.
__network_dim__ = 512 # All network responses have shape = [ __batch_size__, __sequence_dim__, __network_dim__ ]

# Substrate chain block time (seconds).
__blocktime__ = 6

# ---- Config ----
from bittensor._config import config as bit_config

# ---- LOGGING ----
from bittensor._logging import logging

# ---- Protos ----
import bittensor._proto.bittensor_pb2 as proto
import bittensor._proto.bittensor_pb2_grpc as grpc

# ---- Factories -----
from bittensor.utils.balance import Balance
from bittensor._cli import cli
from bittensor._axon import axon
from bittensor._wallet import wallet
from bittensor._receptor import receptor
from bittensor._endpoint import endpoint
from bittensor._dendrite import dendrite
from bittensor._executor import executor
from bittensor._metagraph import metagraph
from bittensor._subtensor import subtensor
from bittensor._tokenizer import tokenizer
from bittensor._serializer import serializer
from bittensor._dataloader import dataloader
from bittensor._receptor import receptor_pool
from bittensor._wandb import wandb
from bittensor._threadpool import prioritythreadpool

# ---- Classes -----
from bittensor._cli.cli_impl import CLI
from bittensor._axon.axon_impl import Axon
from bittensor._config.config_impl import Config
from bittensor._wallet.wallet_impl import Wallet
from bittensor._receptor.receptor_impl import Receptor
from bittensor._endpoint.endpoint_impl import Endpoint
from bittensor._executor.executor_impl import Executor
from bittensor._dendrite.dendrite_impl import Dendrite
from bittensor._metagraph.metagraph_impl import Metagraph
from bittensor._subtensor.subtensor_impl import Subtensor
from bittensor._serializer.serializer_impl import Serializer
from bittensor._dataloader.dataloader_impl import Dataloader
from bittensor._receptor.receptor_pool_impl import ReceptorPool
from bittensor._threadpool.priority_thread_pool_impl import PriorityThreadPoolExecutor

import bittensor.utils.networking as net

# Singluar Neuron instance useful for creating simple miners.
neuron = None

def add_args( parser: argparse.ArgumentParser ):
    """ Accept specific argument from parser
    """
    parser.add_argument('--neuron.use_upnpc', action='store_true', help='''Neuron punches a hole in your router using upnpc''', default=False)
    parser.add_argument('--neuron.use_wandb', action='store_true', help='''Neuron activates its weights and biases powers''', default=False)
    parser.add_argument('--neuron.max_workers', type=int,  help='''Number of maximum threads in the neuron''',default=10)
    logging.add_args( parser )
    wallet.add_args( parser )
    subtensor.add_args( parser )
    metagraph.add_args( parser )
    dataloader.add_args( parser )
    dendrite.add_args( parser )
    axon.add_args( parser )
    wandb.add_args( parser )

def check_config( config ):
    """ Check config for each of the modules 
    """
    logging.check_config( config )
    wallet.check_config( config )
    subtensor.check_config( config )
    metagraph.check_config( config )
    dataloader.check_config( config )
    dendrite.check_config( config )
    axon.check_config( config )

def default_config() -> 'Config':
    """ Return config with checking
    """
    parser = argparse.ArgumentParser()
    add_args( parser )
    bittensor_config = bit_config( parser )
    check_config(bittensor_config)
    return bittensor_config

class Neuron():
    """ Class Neuron which includes all the sub-modules of bittensor
    """

    def __init__(self, 
            config: 'Config',
            root_dir: str = '',
            forward_text: 'Callable' = None,
            backward_text: 'Callable' = None,
            forward_image: 'Callable' = None,
            backward_image: 'Callable' = None,
            forward_tensor: 'Callable' = None,
            backward_tensor: 'Callable' = None,
            blacklist: 'Callable' = None,
        ):
        if config == None: 
            config = default_config()
        self.config = config
        self.root_dir = root_dir
        logging (
            config = self.config,
            logging_dir = root_dir,
        )
        self.wallet = wallet(
            config = self.config
        )
        self.dendrite = dendrite(
            config = self.config,
            wallet = self.wallet
        )
        self.subtensor = subtensor(
            config = self.config
        )
        self.metagraph = metagraph(
            config = self.config,
            subtensor = self.subtensor
        )
        self.axon = axon (
            config = self.config,
            wallet = self.wallet,
            forward_text = forward_text,
            backward_text = backward_text,
            forward_image = forward_image,
            backward_image = backward_image,
            forward_tensor = forward_tensor,
            backward_tensor = backward_tensor,
            blacklist = blacklist,
        )
        self.wandb = None

    def __enter__(self):
        # ---- Setup Wallet. ----
        self.wallet.create()
        try:
            self.metagraph.load().sync().save()
        except Exception:
            self.metagraph.sync().save()
        self.axon.start().subscribe (
            use_upnpc = self.config.neuron.use_upnpc, 
            subtensor = self.subtensor
        )

        # --- Init wandb ----
        if self.config.neuron.use_wandb:
            self.wandb = wandb(
                config = self.config,
                cold_pubkey = self.wallet.coldkeypub,
                hot_pubkey = self.wallet.hotkey.public_key,
                root_dir = self.root_dir
            )
            
    def __exit__ ( self, exc_type, exc_value, exc_traceback ):
        self.axon.stop()
        if self.config.neuron.use_wandb:
            self.wandb.finish()
            
        print(exc_type, exc_value, exc_traceback)

def init( 
        config: 'Config' = None,
        root_dir: str = None,
        forward_text: 'Callable' = None,
        backward_text: 'Callable' = None,
        forward_image: 'Callable' = None,
        backward_image: 'Callable' = None,
        forward_tensor: 'Callable' = None,
        backward_tensor: 'Callable' = None,
        blacklist: 'Callable' = None,
    ) -> Neuron:
    """ Set up global neuron
    """
    global neuron
    neuron = Neuron( 
        config = config,
        root_dir = root_dir,
        forward_text = forward_text,
        backward_text = backward_text,
        forward_image = forward_image,
        backward_image = backward_image,
        forward_tensor = forward_tensor,
        backward_tensor = backward_tensor,
        blacklist = blacklist
    )
    return neuron

# Hardcoded entry point nodes. 
__kusanagi_entrypoints__ = [
    "test.kusanagi.bittensor.com:9944" 
]

__akatsuki_entrypoints__ = [
    "main.akatsuki.bittensor.com:9944"
]

__local_entrypoints__ = [
    '127.0.0.1:9944'
]
