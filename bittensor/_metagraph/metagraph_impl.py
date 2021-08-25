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
import torch
from tqdm import trange

from loguru import logger
from typing import List, Tuple, List

import bittensor
import bittensor.utils.networking as net
import bittensor.utils.weight_utils as weight_utils

class Metagraph( torch.nn.Module ):
    r""" Maintains chain state as a torch.nn.Module.

        Interface:
            tau (:obj:`torch.FloatTensor` of shape :obj:`(1)`): 
                Current, per block, token inflation rate.

            block (:obj:`torch.LongTensor` of shape :obj:`(1)`):
                State block number.

            uids (:obj:`torch.LongTensor` of shape :obj:`(metagraph.n)`):
                UIDs for each neuron.
            
            stake (:obj:`torch.LongTensor` of shape :obj:`(metagraph.n)`):
                Stake balance for each neuron ordered by uid.
                
            lastemit (:obj:`torch.LongTensor` of shape :obj:`(metagraph.n)`):
                Last emission call for each neuron ordered by uid.

            weights (:obj:`torch.FloatTensor` of shape :obj:`(metagraph.n, metagraph.n)`):
                Full weight matrix on chain ordered by uid.

            neurons (:obj:`torch.LongTensor` of shape :obj:`(metagraph.n, -1)`) 
                Tokenized endpoint information.

    """
    def __init__( self, subtensor ):
        r""" Initializes a new Metagraph torch chain interface object.
        """
        super(Metagraph, self).__init__()
        self.subtensor = subtensor
        self.version = torch.nn.Parameter( torch.tensor( [ bittensor.__version_as_int__ ], dtype=torch.int64), requires_grad=False )
        self.n = torch.nn.Parameter( torch.tensor( [0], dtype=torch.int64), requires_grad=False )
        self.tau = torch.nn.Parameter( torch.tensor( [0.5], dtype=torch.float32), requires_grad=False )
        self.block = torch.nn.Parameter( torch.tensor( [0], dtype=torch.int64), requires_grad=False )
        self.uids = torch.nn.Parameter( torch.tensor( [], dtype=torch.int64), requires_grad=False )

        self.stake = torch.nn.Parameter( torch.tensor( [], dtype=torch.float32), requires_grad=False )
        self.ranks = torch.nn.Parameter( torch.tensor( [], dtype=torch.float32), requires_grad=False )
        self.trust = torch.nn.Parameter( torch.tensor( [], dtype=torch.float32), requires_grad=False )
        self.incentive = torch.nn.Parameter( torch.tensor( [], dtype=torch.float32), requires_grad=False )
        self.inflation = torch.nn.Parameter( torch.tensor( [], dtype=torch.float32), requires_grad=False )
        self.dividends = torch.nn.Parameter( torch.tensor( [], dtype=torch.float32), requires_grad=False )
        
        self.lastupdate = torch.nn.Parameter( torch.tensor( [], dtype=torch.int64), requires_grad=False )
        self.weights = torch.nn.Parameter( torch.tensor( [], dtype=torch.int64), requires_grad=False )
        self.bonds = torch.nn.Parameter( torch.tensor( [], dtype=torch.int64), requires_grad=False )
        self.endpoints = torch.nn.Parameter( torch.tensor( [], dtype=torch.int64), requires_grad=False )
        self.balances = torch.nn.Parameter( torch.tensor( [], dtype=torch.float32), requires_grad=False )
        self._endpoint_objs = None

    @property
    def S(self) -> torch.FloatTensor:
        r""" Returns neurons stake values.
             
             Returns:
                S (:obj:`torch.FloatTensor` of shape :obj:`(metagraph.n)`):
                    Stake of each known neuron.
        """
        return self.stake

    @property
    def R(self) -> torch.FloatTensor:
        r""" Returns neuron ranks: W^t * S
             
             Returns:
                rank (:obj:`torch.FloatTensor` of shape :obj:`(metagraph.n)`):
                    Rank of each neuron.
        """
        return self.ranks

    @property
    def T(self) -> torch.FloatTensor:
        r""" Returns neurons trust scores
             
             Returns:
                T (:obj:`torch.FloatTensor` of shape :obj:`(metagraph.n)`):
                    Trust scores for each neuron measured in tao.
        """
        return self.trust

    @property
    def I(self) -> torch.FloatTensor:
        r""" Returns neuron inflation: tau * I
        
            Returns:
                inflation (:obj:`torch.FloatTensor` of shape :obj:`(metagraph.n)`):
                    Inflation in tao for each neuron.
        """
        return self.inflation

    @property
    def D(self) -> torch.FloatTensor:
        r""" Returns neuron dividends
        
            Returns:
                dividends (:obj:`torch.FloatTensor` of shape :obj:`(metagraph.n)`):
                    Dividends for each neuron measured in tao.
        """
        return self.dividends

    @property
    def B(self) -> torch.FloatTensor:
        r""" Returns neuron bonds
            Returns:
                B (:obj:`torch.FloatTensor` of shape :obj:`(metagraph.n, metagraph.n)`):
                    Bond matrix.
        """
        return self.bonds

    @property
    def W(self) -> torch.FloatTensor:
        r""" Return full weight matrix from chain.
             Returns:
                W (:obj:`torch.FloatTensor` of shape :obj:`(metagraph.n, metagraph.n)`):
                    Weight matrix.
        """
        return self.weights

    @property
    def hotkeys( self ) -> List[str]:
        r""" Returns hotkeys for each neuron.
            Returns:
                hotkeys (:obj:`List[str] of shape :obj:`(metagraph.n)`):
                    Neuron hotkeys.
        """
        if self.n.item() == 0:
            return []
        return [ neuron.hotkey for neuron in self.endpoint_objs ]

    @property
    def coldkeys( self ) -> List[str]:
        r""" Returns coldkeys for each neuron.
            Returns:
                coldkeys (:obj:`List[str] of shape :obj:`(metagraph.n)`):
                    Neuron coldkeys.
        """
        if self.n.item() == 0:
            return []
        return [ neuron.coldkey for neuron in self.endpoint_objs ]

    @property
    def modalities( self ) -> List[str]:
        r""" Returns the modality for each neuron.
            Returns:
                coldkeys (:obj:`List[str] of shape :obj:`(metagraph.n)`):
                    Neuron coldkeys.
        """
        if self.n.item() == 0:
            return []
        return [ neuron.modality for neuron in self.endpoint_objs ]

    @property
    def addresses( self ) -> List[str]:
        r""" Returns ip addresses for each neuron.
            Returns:
                coldkeys (:obj:`List[str] of shape :obj:`(metagraph.n)`):
                    Neuron address.
        """
        if self.n.item() == 0:
            return []
        return [ net.ip__str__( neuron.ip_type, neuron.ip, neuron.port ) for neuron in self.endpoint_objs ]

    @property
    def endpoint_objs( self ) -> List['bittensor.Endpoint']:
        r""" Returns endpoints as objects.
            Returns:
                endpoint_obj (:obj:`List[bittensor.Endpoint] of shape :obj:`(metagraph.n)`):
                    Endpoints as objects.
        """
        if self.n.item() == 0:
            return []
        elif self._endpoint_objs != None:
            return self._endpoint_objs
        else:
            self._endpoint_objs = [ bittensor.endpoint.from_tensor( tensor ) for tensor in self.endpoints ]
            return self._endpoint_objs

    def clear( self ) -> 'Metagraph':
        r""" Erases Metagraph state.
        """
        self.n = torch.nn.Parameter( torch.tensor( [0], dtype=torch.int64), requires_grad=False )
        self.tau = torch.nn.Parameter( torch.tensor( [0.5], dtype=torch.float32), requires_grad=False )
        self.block = torch.nn.Parameter( torch.tensor( [0], dtype=torch.int64), requires_grad=False )
        self.uids = torch.nn.Parameter( torch.tensor( [], dtype=torch.int64), requires_grad=False )
        self.stake = torch.nn.Parameter( torch.tensor( [], dtype=torch.float32), requires_grad=False )
        self.ranks = torch.nn.Parameter( torch.tensor( [], dtype=torch.float32), requires_grad=False )
        self.trust = torch.nn.Parameter( torch.tensor( [], dtype=torch.float32), requires_grad=False )
        self.incentive = torch.nn.Parameter( torch.tensor( [], dtype=torch.float32), requires_grad=False )
        self.inflation = torch.nn.Parameter( torch.tensor( [], dtype=torch.float32), requires_grad=False )
        self.dividends = torch.nn.Parameter( torch.tensor( [], dtype=torch.float32), requires_grad=False )
        self.lastupdate = torch.nn.Parameter( torch.tensor( [], dtype=torch.int64), requires_grad=False )
        self.bonds = torch.nn.Parameter( torch.tensor( [], dtype=torch.int64), requires_grad=False )
        self.weights = torch.nn.Parameter( torch.tensor( [], dtype=torch.int64), requires_grad=False )
        self.endpoints = torch.nn.Parameter( torch.tensor( [], dtype=torch.int64), requires_grad=False )
        self.balances = torch.nn.Parameter( torch.tensor( [], dtype=torch.float32), requires_grad=False )
        self._endpoint_objs = None
        return self

    def load( self, network:str = None  ) -> 'Metagraph':
        r""" Loads this metagraph object's state_dict from bittensor root dir.
            Args: 
                network: (:obj:`str`, required):
                    Name of state_dict to load, defaults to kusanagi
        """
        try:
            if network == None:
                network = self.subtensor.network
            metagraph_path = '~/.bittensor/' + str(network) + '.pt'
            metagraph_path = os.path.expanduser(metagraph_path)
            if os.path.isfile(metagraph_path):
                self.load_from_path( path = metagraph_path )
            else:
                logger.warning('Did not load metagraph from path: {}, file does not exist. Run metagraph.save() first.', metagraph_path)
        except Exception as e:
            logger.exception(e)
        return self

    def save( self, network:str = None ) -> 'Metagraph':
        r""" Saves this metagraph object's state_dict under bittensor root dir.
            Args: 
                network: (:obj:`str`, required):
                    Name of state_dict, defaults to kusanagi
        """
        if network == None:
            network = self.subtensor.network
        return self.save_to_path( path = '~/.bittensor/', filename = str(network) + '.pt')

    def load_from_path(self, path:str ) -> 'Metagraph':
        r""" Loads this metagraph object with state_dict under the specified path.
            Args: 
                path: (:obj:`str`, required):
                    Path to load state_dict.
        """
        full_path = os.path.expanduser(path)
        metastate = torch.load( full_path )
        return self.load_from_state_dict( metastate )

    def save_to_path(self, path:str, filename:str ) -> 'Metagraph':
        r""" Saves this metagraph object's state_dict to the specified path.
            Args: 
                path: (:obj:`str`, required):
                    Path to save state_dict.
        """
        full_path = os.path.expanduser(path)
        os.makedirs(full_path, exist_ok=True)
        metastate = self.state_dict()
        torch.save(metastate, full_path + '/' + filename)
        return self

    def load_from_state_dict(self, state_dict:dict ) -> 'Metagraph':
        r""" Loads this metagraph object from passed state_dict.
            Args: 
                state_dict: (:obj:`dict`, required):
                    Metagraph state_dict. Must be same as that created by save_to_path.
        """
        self.version = torch.nn.Parameter( state_dict['version'], requires_grad=False )
        self.n = torch.nn.Parameter( state_dict['n'], requires_grad=False )
        self.tau = torch.nn.Parameter( state_dict['tau'], requires_grad=False )
        self.block = torch.nn.Parameter( state_dict['block'], requires_grad=False )
        self.uids = torch.nn.Parameter( state_dict['uids'], requires_grad=False )
        self.stake = torch.nn.Parameter( state_dict['stake'], requires_grad=False )
        self.trust = torch.nn.Parameter( state_dict['trust'], requires_grad=False )
        self.ranks = torch.nn.Parameter( state_dict['ranks'], requires_grad=False )
        self.incentive = torch.nn.Parameter( state_dict['incentive'], requires_grad=False )
        self.inflation = torch.nn.Parameter( state_dict['inflation'], requires_grad=False )
        self.dividends = torch.nn.Parameter( state_dict['dividends'], requires_grad=False )
        self.lastupdate = torch.nn.Parameter( state_dict['lastupdate'], requires_grad=False )
        self.bonds = torch.nn.Parameter( state_dict['bonds'], requires_grad=False )
        self.weights = torch.nn.Parameter( state_dict['weights'], requires_grad=False )
        self.endpoints = torch.nn.Parameter( state_dict['endpoints'], requires_grad=False )
        self.balances = torch.nn.Parameter( state_dict['balances'], requires_grad=False )
        self._endpoint_objs = None
        return self

    def sync ( self, block: int = None ) -> 'Metagraph':
        r""" Synchronizes this metagraph with the chain state.
        """

        # Query chain info.
        for i in trange(13):
            if i == 0:
                #chain_lastupdate = dict( self.subtensor.get_lastupdate(block=block) ) #  Optional[ List[Tuple[uid, lastemit]] ]
                pass
            if i == 1:
                chain_stake = dict( self.subtensor.get_stake(block=block) ) #  Optional[ List[Tuple[uid, stake]] ]
            if i == 2:
                chain_block = block if block != None else int( self.subtensor.get_current_block()) #  Optional[ int ]
            if i == 3:
                chain_weights_uids = dict ( self.subtensor.get_weight_uids(block=block) )
            if i == 4:
                chain_weights_vals = dict ( self.subtensor.get_weight_vals(block=block) )
            if i == 5:
                chain_endpoints = dict ( self.subtensor.neurons(block=block) )
            if i == 6:
                chain_balances = self.subtensor.get_balances(block = block)
            if i == 7:
                chain_ranks = self.subtensor.get_ranks(block = block)
            if i == 8:
                chain_trust = self.subtensor.get_trust(block = block)
            if i == 9:
                chain_incentive = self.subtensor.get_incentive(block = block)
            if i == 10:
                chain_inflation = self.subtensor.get_inflation(block = block)
            if i == 11:
                chain_dividends = self.subtensor.get_dividends(block = block)
            if i == 12:
                chain_bonds = self.subtensor.get_bonds(block = block)

        # Build state.
        size = len(chain_stake)
        new_n = torch.tensor( [size], dtype=torch.int64)
        new_block = torch.tensor( [chain_block], dtype=torch.int64)
        new_uids = torch.tensor( range(size) ,  dtype=torch.int64)
        new_stake = torch.tensor([ float(chain_stake[uid])/float(1000000000) for uid in range( size ) ], dtype=torch.float32)
        new_ranks = torch.tensor([ float(chain_ranks[uid])/float(1000000000) for uid in range( size ) ], dtype=torch.float32)
        new_trust = torch.tensor([ float(chain_trust[uid])/float(1000000000) for uid in range( size ) ], dtype=torch.float32)
        new_incentive = torch.tensor([ float(chain_incentive[uid])/float(1000000000) for uid in range( size ) ], dtype=torch.float32)
        new_inflation = torch.tensor([ float(chain_inflation[uid])/float(1000000000) for uid in range( size ) ], dtype=torch.float32)
        new_dividends = torch.tensor([ float(chain_dividends[uid])/float(1000000000) for uid in range( size ) ], dtype=torch.float32)
        #new_lastupdate = torch.tensor( [chain_lastupdate[uid] for uid in range( size ) ], dtype=torch.int64)

        # Set params.
        self.n = torch.nn.Parameter( new_n, requires_grad=False )
        self.block = torch.nn.Parameter( new_block, requires_grad=False )
        self.uids = torch.nn.Parameter( new_uids, requires_grad=False )
        self.stake = torch.nn.Parameter( new_stake, requires_grad=False )
        self.ranks = torch.nn.Parameter( new_ranks, requires_grad=False )
        self.trust = torch.nn.Parameter( new_trust, requires_grad=False )
        self.incentive = torch.nn.Parameter( new_incentive, requires_grad=False )
        self.inflation = torch.nn.Parameter( new_inflation, requires_grad=False )
        self.dividends = torch.nn.Parameter( new_dividends, requires_grad=False )
        #self.lastupdate = torch.nn.Parameter( new_lastupdate, requires_grad=False )
        self.weights = torch.nn.Parameter( torch.zeros( [new_n, new_n] , dtype=torch.float32), requires_grad=False )
        self.bonds = torch.nn.Parameter( torch.zeros( [new_n, new_n] , dtype=torch.float32), requires_grad=False )
        self.endpoints = torch.nn.Parameter( torch.zeros( [new_n, 250] , dtype=torch.int64), requires_grad=False )
        self.balances = torch.nn.Parameter( torch.zeros( [new_n] , dtype=torch.float32), requires_grad=False )
        self._endpoint_objs = []

        # Fill values for weights and endpoint information.
        for uid in range( size ):
             # Fill row from weights.
            row_weights = weight_utils.convert_weight_uids_and_vals_to_tensor( size, chain_weights_uids[uid], chain_weights_vals[uid] )
            self.weights[uid] = row_weights

            # Fill bonds from query.
            row_bonds = [ 0 for _ in range( new_n )]
            for uid_j, val in chain_bonds[uid]:
                row_bonds[uid_j] = float(val)/float(1000000000.0)
            self.bonds[uid] = torch.tensor( row_bonds, dtype=torch.float32 ) 

            # Fill Neuron info.
            endpoint_obj = bittensor.endpoint.from_dict( chain_endpoints[uid] )
            endpoint_tensor = endpoint_obj.to_tensor()
            self.endpoints[uid] = endpoint_tensor
            self._endpoint_objs.append( endpoint_obj )

        # Fill balances.
        for cold in chain_balances.keys():
            try:
                uid = self.coldkeys.index( cold )
                self.balances[uid] = chain_balances[cold]
            except:
                pass
            
        # For contructor.
        return self
    
    def __str__(self):
        return "Metagraph({}, {}, {})".format(self.n.item(), self.block.item(), self.subtensor.network)
        
    def __repr__(self):
        return self.__str__()
        