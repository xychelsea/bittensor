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

""" Template server.

Example:
    $ import neurons
    $ neurons.text.template_server.neuron().run()
"""

import bittensor
import os

from .nucleus_impl import server
from .run import serve

class neuron:
    r"""
    Creates a bittensor neuron that specializes in the serving. The template server miner
    serves a NLP model from huggingface on the bittensor network. By default, the model does 
    not train itself and thus requires less memory to run. 

    Args: 
            config (:obj:`bittensor.Config`, `optional`): 
                bittensor.server.config()
            subtensor (:obj:bittensor.subtensor , `optional`):
                bittensor subtensor connection
            wallet (:obj:bittensor.wallet, `optional`):
                bittensor wallet object
            axon (:obj:bittensor.axon, `optional`):
                bittensor axon object
            metagraph (:obj:bittensor.metagraph, `optional`):
                bittensor metagraph object

    Examples:: 
            >>> subtensor = bittensor.subtensor(network='nakamoto')
            >>> server = bittensor.neuron.text.template_server.neuron(subtensor=subtensor)
            >>> server.run()
    """
    def __init__(
        self, 
        config: 'bittensor.config' = None,
        subtensor: 'bittensor.subtensor' = None,
        wallet: 'bittensor.wallet' = None,
        axon: 'bittensor.axon' = None,
        metagraph: 'bittensor.metagraph' = None,

    ):
        if config == None: config = server.config()
        config = config; 
        self.check_config( config )
        bittensor.logging (
            config = config,
            logging_dir = config.neuron.full_path,
        )

        self.model = server(config = config)
        self.config = config

        self.subtensor = subtensor
        self.wallet = wallet
        self.axon = axon
        self.metagraph = metagraph

    def run(self):
        serve(
            self.config,
            self.model,
            subtensor = self.subtensor,
            wallet = self.wallet,
            axon = self.axon,
            metagraph= self.metagraph,
        )


    @classmethod
    def config(cls):
        return server.config()

    @staticmethod
    def check_config( config: 'bittensor.Config' ):
        r""" Checks/validates the config namespace object.
        """
        bittensor.logging.check_config( config )
        bittensor.wallet.check_config( config )
        bittensor.subtensor.check_config( config )
        bittensor.metagraph.check_config( config )
        bittensor.dataset.check_config( config )
        bittensor.axon.check_config( config )
        bittensor.wandb.check_config( config )
        full_path = os.path.expanduser('{}/{}/{}/{}'.format( config.logging.logging_dir, config.wallet.get('name', bittensor.defaults.wallet.name), config.wallet.get('hotkey', bittensor.defaults.wallet.hotkey), config.neuron.name ))
        config.neuron.full_path = os.path.expanduser(full_path)
        if not os.path.exists(config.neuron.full_path):
            os.makedirs(config.neuron.full_path)
