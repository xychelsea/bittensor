#!/bin/python3
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
""" The Exodus miner.

Example:
    $ python miners/text/gpt2_exodus.py

"""

import argparse
import bittensor
import math
import onnx
import torch
import traceback
import os
import sys
import yaml
import wandb

from termcolor import colored
from typing import List
from qqdm import qqdm, format_str
from datetime import datetime
from loguru import logger; logger = logger.opt(colors=True)
from types import SimpleNamespace
from nuclei.gpt2 import GPTPooler,Block,GPTConfig
from torch.nn.utils import clip_grad_norm_
from tensorboard import program
from torch.utils.tensorboard import SummaryWriter
import bittensor.utils.networking as net
import torch.nn as nn
from typing import Callable
from torch.nn import functional as F

class GPT2Nucleus(torch.nn.Module):

    def __init__(self, config: bittensor.Config = None, routing_callback = None, **kwargs):
        """The full GPT language model, with context of a block size.
            Args:
                config (:obj: `bittensor.Config`, `required`):
                    munched config class.
        """
        super(GPT2Nucleus, self).__init__()
        if config == None: config = GPT2Nucleus.config()
        GPT2Nucleus.check_config( config )
        self.config = config.nucleus
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # To be set.
        self.routing_callback = routing_callback
        
        self.vocab_size = bittensor.__vocab_size__
        self.n_embd = bittensor.__network_dim__

        gpt_config = GPTConfig(
            vocab_size = bittensor.__vocab_size__,
            n_embd=bittensor.__network_dim__,
            n_head=self.config.n_head,
            n_layer=self.config.n_layer,
            block_size=self.config.block_size,
            embd_pdrop=self.config.embd_pdrop,
            resid_pdrop=self.config.resid_pdrop,
            attn_pdrop=self.config.attn_pdrop
        )
        # Token embedding layer. 
        # [bittensor.__vocab_size__, bittensor.__network_dim__]
        self.tok_emb = nn.Embedding(self.vocab_size, self.n_embd)

        # Positional embedding.
        # [1, block_size, bittensor.__network_dim__]
        self.pos_emb = nn.Parameter(torch.zeros(1, self.config.block_size, self.n_embd))
        self.drop = nn.Dropout(self.config.embd_pdrop)

        # Transformer blocks
        self.blocks = nn.Sequential(*[Block(gpt_config) for _ in range(self.config.n_layer)])

        # Decoder head
        self.ln_f = nn.LayerNorm(self.n_embd)

        # Head
        # [ bittensor.__network_dim__, bittensor.__network_dim__ ]
        self.head = nn.Linear(self.n_embd, self.n_embd, bias=False)

        # pooler_layer: pools the hidden units for use by the pkm dendrite rpc query.
        self.pooler = GPTPooler(gpt_config)

        # Hidden layer
        self.hidden_layer = nn.Linear( self.n_embd, self.n_embd)

        # Target layer
        self.target_layer = nn.Linear(self.n_embd, self.vocab_size, bias=False )

        # Block size here corresponds to sequence lengths
        self.block_size = self.config.block_size
        self.apply(self._init_weights)

        # Loss function: MLM cross-entropy loss.
        # predicted: [batch_size, sequence_len, 1], targets: [batch_size, sequence_len, 1] -> [1]
        self.loss_fct = nn.CrossEntropyLoss()
                
        self.num_parameters = sum(p.numel() for p in self.parameters())
    
    @staticmethod   
    def config() -> 'bittensor.Config':
        parser = argparse.ArgumentParser()
        GPT2Nucleus.add_args( parser )
        return bittensor.config( parser )

    @staticmethod
    def add_args( parser: argparse.ArgumentParser ):
        parser.add_argument('--nucleus.n_head',  default=32, type=int, help='Number of attention heads for each attention layer in the Transformer encoder.')
        parser.add_argument('--nucleus.n_layer', default=12, type=int, help='Number of hidden layers in the Transformer encoder.')
        parser.add_argument('--nucleus.block_size', default=20, type=int, help='Number of hidden layers in the Transformer encoder.')
        parser.add_argument('--nucleus.embd_pdrop', default=0.1, type=float, help='GPT embedding dropout probability.')
        parser.add_argument('--nucleus.resid_pdrop', default=0.1, type=float, help='GPT residual dropout probability.')
        parser.add_argument('--nucleus.attn_pdrop', default=0.1, type=float, help='GPT attention dropout probability.')

    @staticmethod
    def check_config( config: 'bittensor.Config' ):
        assert config.nucleus

    def attach(self, servicer: object ):
        """ Attaches the passed servicer's routing function to this nucleus.

            Returns:
                servicer (:callabl:`Callable[ [torch.Tensor, torch.Tensor], torch.Tensor `, `required`): 
                    servicer implementing function route()
        """
        self.attach_routing_callback( servicer.remote )

    def attach_routing_callback(self, routing_callback: Callable[ [torch.Tensor, torch.Tensor], torch.Tensor ] ):
        """ Assigns the passed routing_callback to this nucleus.

            Returns:
                routing_callback (:callabl:`Callable[ [torch.Tensor, torch.Tensor], torch.Tensor `, `required`): 
                    Routing function to call on self.route()
        """
        # TODO(const): type checking.
        self.routing_callback = routing_callback


    def route( self, inputs: torch.Tensor, query: torch.Tensor ) -> torch.FloatTensor:
        """ Calls this nucleus's subscribed routing function. self.routing_callback must be set before this call is made.

        Args:
            inputs (:obj:`torch.int64` of shape :obj:`( batch_size, sequence_len )`, `required`): 
                    Batch_size length list of tokenized sentences.

            query (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, query_dimension)`, `required`): 
                    Context tensor used to select which neurons to query for each example.
            
        Returns:
            remote_context (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_len, bittensor.__network_dim__)`, `required`): 
                joined responses from network call.
        """
        if self.routing_callback == None:
            raise RuntimeError('The routing function must be set on this nucleus before a remote_forward call can execute.')
        else:
            return self.routing_callback( inputs = inputs, query = query )

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)

            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
            
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    

    def forward_text(self, inputs: torch.int64):
        """ Local forward inputs through the CLM GPT Nucleus.

            Args:
                inputs (:obj:`torch.int64` of shape :obj:`(batch_size, sequence_len)`, `required`): 
                    Batch_size length list of tokenized sentences.
            
            Returns:
                hidden (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_len, bittensor.__network_dim__)`, `required`): 
                    Hidden layer representation produced using the local_context.
        """
        # Truncate seq length of incoming inputs if they are too long
        initial_length = inputs.size(1)
        inputs = inputs if initial_length <= self.block_size else inputs[:, -self.block_size:]
        hidden = self.local_forward(inputs=inputs.to(self.device), training = False).local_hidden
        
        # Now pad the output tensor back to the original length
        if initial_length > self.block_size:
            diff = initial_length - self.block_size
            padding = (0,0,diff,0)
            hidden = torch.nn.functional.pad(hidden, padding, "constant", 0)
            
        return hidden


    def local_forward(self, inputs: torch.int64, training : bool = True) -> SimpleNamespace:
        """ Forward pass through GPT2 nucleus.

            Args:
                inputs (:obj:`torch.int64` of shape :obj:`(batch_size, block_size)`, `required`): 
                    Batch_size length x list of text sentences.

                training (:obj:`bool')`, `optional`, defaults to True):
                    Switch to True if this forward pass computes a CLM loss.

            SimpleNamespace {
                    local_context (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_len, bittensor.__network_dim__)`, `required`):
                        Hidden layer context.

                    local_hidden (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_len, bittensor.__network_dim__)`, `required`):
                        Hidden layer encoding produced using local_context.

                    local_target (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_len, bittensor.__vocab_size__)`, `optional`):
                        GPT MLM Target predictions produced using local_context. 

                    local_target_loss (:obj:`torch.FloatTensor` of shape :obj:`(1)`, `optional`): 
                        GPT MLM loss using local_context.
                }
        """
        _, t = inputs.size()
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."

        # FWD locally
        # Each index maps to a learnable vector
        token_embeddings = self.tok_emb(inputs) 
        # Each Position maps to a learnable vector
        position_embeddings = self.pos_emb[:, :t, :]

        output = SimpleNamespace()
        # Dropout on token embeddings and position embeddings
        out = self.drop(token_embeddings + position_embeddings)

        # 
        out = self.blocks(out)
        out = self.ln_f(out)
        output.local_context = self.head(out)
        output.local_hidden = self.hidden_layer(output.local_context)

        if training :
            output.local_target = self.target_layer(output.local_hidden)
            shift_logits = output.local_target[..., :-1, :].contiguous()
            shift_labels = inputs[..., 1:].contiguous()            
            output.local_target_loss = self.loss_fct( shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1) )
            
        return output


    def remote_forward(self, inputs: torch.int64, training: bool) -> SimpleNamespace:
        """ Forward pass inputs and labels through the GPT2 module and into the remote network.


        Args:
            inputs (:obj:`torch.int64` of shape :obj:`(batch_size, sequence_len)`, `required`): 
                    Batch_size length list of text sentences.

            training (:obj:`bool')`, `optional`, defaults to True):
                Switch to True if this forward pass computes an MLM loss.

        Returns:
            self.local_forward() + SimpleNamespace ( 

                    remote_hidden (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_len, bittensor.__network_dim__)`, `optional`): 
                        Hidden layer encoding produced using the remote_context.

                    remote_target (:obj:`torch.FloatTensor` of shape :obj:`(batch_size,  bittensor.__vocab_size__)`, `optional`):
                        GPT MLM Target predictions using the remote_context.

                    remote_target_loss (:obj:`torch.FloatTensor` of shape :obj:`(1)`, `optional`):
                        GPT MLM loss using the remote_context.

                    distillation_loss (:obj:`torch.FloatTensor` of shape :obj:`(1)`, `optional`): 
                        Distillation loss between local_context and remote_context.
            )
        """
        # Send inputs to appropriate device
        inputs = inputs.to(self.device)

        inputs = torch.clamp(inputs, 0, bittensor.__vocab_size__) # Filter out of range tokens.
        # Run local model
        # output = SimpleNamespace
        output = self.local_forward(inputs, training)
        
        # pooled: pooled hidden layer from local run, used as query context
        #print(output.local_hidden.detach().shape)
        pooled = self.pooler(output.local_hidden.detach())
        #print(pooled.shape)

        # remote_context: joined responses from a dendrite.forward_text call.
        # remote_context.shape = [batch_size, sequence_len (or block_size), bittensor.__network_dim__]
        output.remote_context = self.route( inputs = inputs.to(self.device), query = pooled )
        
        # distillation_loss : distillation loss between local_context and remote_context
        # distillation_loss.shape = [1]
        output.distillation_loss = F.mse_loss(output.local_context, output.remote_context.detach())

        # remote_hidden: hidden l;ayer encoding using remote_context.
        # remote_hidden.shape = [batch_size, sequence_len, bittensor.__network_dim__]
        output.remote_hidden = self.hidden_layer( output.remote_context )

        if training:
            # remote_target : projection of remote_hidden onto the target dimension
            # remote_target.shape = [batch_size, sequence_len, bittensor.__vocab_size__]
            output.remote_target = self.target_layer(output.remote_hidden)

            # remote_target_loss : CLM loss between remote_target and passed_targets.
            # remote_target_loss.shape = [1]
            shift_logits = output.remote_target[..., :-1, :].contiguous()
            shift_labels = inputs[..., 1:].contiguous()


            output.remote_target_loss = self.loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            shift_labels.view(-1)
        
        return output




class neuron:

    def __init__( self, config: 'bittensor.config' = None ):
        r""" Initializes a neuron with the passed config.
        """
        wandb.login()
        if config == None: config = neuron.config()
        self.config = config; neuron.check_config( self.config ); print ( self.config )
        bittensor.logging (
            config = self.config,
            logging_dir = self.config.neuron.full_path,
        )
        self.device = torch.device(
            device = self.config.neuron.device
        )
        self.wallet = bittensor.wallet(
            config = self.config
        )
        self.dendrite = bittensor.dendrite(
            config = self.config,
            wallet = self.wallet
        )
        self.subtensor = bittensor.subtensor(
            config = self.config
        )
        self.metagraph = bittensor.metagraph(
            config = self.config
        )
        self.axon = bittensor.axon (
            config = self.config,
            wallet = self.wallet,
            forward_callback = self.forward,
            backward_callback = self.backward
        )
        self.dataset = bittensor.dataloader (
            config = self.config
        )
        self.nucleus = GPT2Nucleus(
            config = self.config,
            routing_callback = self.remote
        ).to( self.device )
        self.optimizer = torch.optim.SGD(
            [
                {"params": self.nucleus.parameters()}
            ],
            lr = self.config.neuron.learning_rate,
            weight_decay = self.config.neuron.weight_decay,
        )
        self.tensorboard = SummaryWriter(
            log_dir = self.config.neuron.tensorboard_dir
        )
        self.mechanism_weights = torch.ones( [0] )
        self.epoch = 0
        self.global_step = 0
        self.epoch_loss = math.inf/2
        self.best_epoch_loss = math.inf
        self.topk_uids = []

    @staticmethod
    def config() -> 'bittensor.Config':
        r""" Fills a config namespace object with defaults or information from the command line.
        """
        parser = argparse.ArgumentParser()
        parser.add_argument('--neuron.config', type=str, help='If set, defaults are overridden by passed file.')
        parser.add_argument('--neuron.modality', type=int, help='''Miner network modality. TEXT=0, IMAGE=1. Currently only allowed TEXT''', default=0)
        parser.add_argument('--neuron.use_upnpc', action='store_true', help='''Turns on port forwarding on your router using upnpc.''', default=False)
        parser.add_argument('--neuron.use_tensorboard', action='store_true', help='Turn on bittensor logging to tensorboard', default=True)
        parser.add_argument('--neuron.learning_rate', type=float, help='Training initial learning rate.', default=3e-2)
        parser.add_argument('--neuron.weight_decay', type=float, help='nucleus parameter weight decay.', default=0.25)
        parser.add_argument('--neuron.clip_gradients', type=float, help='Implement gradient clipping to avoid exploding loss on smaller architectures.', default=1.0)
        parser.add_argument('--neuron.n_epochs', type=int, help='Number of training epochs.', default=sys.maxsize )
        parser.add_argument('--neuron.epoch_length', type=int, help='Iterations of training per epoch', default=500)
        parser.add_argument('--neuron.batch_size_train', type=int, help='Training batch size.', default=2)
        parser.add_argument('--neuron.reload', action='store_true', help='''Reload training from previous trial run.''', default=False )
        parser.add_argument('--neuron.restart_on_failure',  action='store_true', help='''Restart miner on unknown error.''', default=False)
        parser.add_argument('--neuron.compute_remote_gradients', action='store_true', help='''Does the neuron compute and return gradients from backward queries.''', default=False)
        parser.add_argument('--neuron.accumulate_remote_gradients', action='store_true', help='''Does the neuron accumulate remote gradients from backward queries.''', default=False)
        parser.add_argument('--neuron.name', type=str, help='Trials for this miner go in miner.root / (wallet_cold - wallet_hot) / miner.name ', default='gpt2_exodus')
        parser.add_argument('--neuron.device', type=str, help='Neuron default training device cpu/cuda', default=("cuda" if torch.cuda.is_available() else "cpu"))
        parser.add_argument('--neuron.wandb',type=str, help='Weights and Bias setting (online or offline)', default='offline')
        parser.add_argument('--neuron.sparsity',type=int, help='remote sparsity', default=20)
        bittensor.logging.add_args( parser )
        bittensor.wallet.add_args( parser )
        bittensor.subtensor.add_args( parser )
        bittensor.metagraph.add_args( parser )
        bittensor.dataloader.add_args( parser )
        bittensor.dendrite.add_args( parser )
        bittensor.axon.add_args( parser )
        GPT2Nucleus.add_args( parser )
        
        config_file_path = vars(parser.parse_known_args()[0])['neuron.config']
        if config_file_path:
            #loads config_file and updates defaults
            config_file_path = os.path.expanduser(config_file_path)
                
            try:
                with open(config_file_path) as f:
                    params_config = yaml.safe_load(f) 
                    print('Config File Detected at {} updating defaults'.format(config_file_path))
                    parser.set_defaults(**params_config)
                    
            except Exception as e:
                print('Error in loading: {} using default parser settings'.format(e))

        return bittensor.config( parser )

    @staticmethod
    def check_config( config: 'bittensor.Config' ):
        r""" Checks/validates the config namespace object.
        """
        assert config.neuron.batch_size_train > 0, "batch_size_train must be a positive value"
        assert config.neuron.learning_rate > 0, "learning_rate must be a positive value."
        bittensor.logging.check_config( config )
        bittensor.wallet.check_config( config )
        bittensor.subtensor.check_config( config )
        bittensor.metagraph.check_config( config )
        bittensor.dataloader.check_config( config )
        bittensor.dendrite.check_config( config )
        bittensor.axon.check_config( config )
        GPT2Nucleus.check_config( config )
        full_path = os.path.expanduser('{}/{}/{}'.format( config.logging.logging_dir, config.wallet.name + "-" + config.wallet.hotkey, config.neuron.name ))
        config.neuron.full_path = os.path.expanduser(full_path)
        config.neuron.tensorboard_dir = config.neuron.full_path + '/tensorboard-' + '-'.join(str(datetime.now()).split())
        if not os.path.exists(config.neuron.full_path):
            os.makedirs(config.neuron.full_path)

    def __enter__(self):
        self.startup()

    def __exit__ ( self, exc_type, exc_value, exc_traceback ):
        self.shutdown()

    def run( self ):
        r""" Miner main loop.
        """
        with wandb.init(
            project='GPT-miner',
            dir = self.config.neuron.full_path,
            config = self.config,
            mode = self.config.neuron.wandb
        ):
            # ---- Startup/Shutdown ----
            with self:
                
                # ---- Optionally reload from previous run ----
                if self.config.neuron.reload:
                    self.reload()
                else:
                    self.checkpoint()
                    
                wandb.watch(self.nucleus.blocks,self.nucleus.loss_fct,log ='all',log_freq=10,idx=0)
                # --- Run until n_epochs ----
                while self.epoch < self.config.neuron.n_epochs:
                    try:
                        wandb.log({'Epoch':self.epoch})
                        # ---- Train state ----
                        self.run_epoch()

                        # ---- Set weights on chain ----
                        self.set_mechanism_weights()

                        # ---- Checkpoint state ----
                        self.checkpoint()

                    except KeyboardInterrupt:
                        # --- User ended session ----
                        break

                    except Exception as e:
                        # --- Unknown error ----
                        logger.exception('Unknown exception: {} with traceback {}', e, traceback.format_exc())
                        if self.config.neuron.restart_on_failure == True:
                            logger.info('Restarting from last saved state.')
                            self.reload()
                        else:
                            break

    # --- Run Epoch ----
    def run_epoch( self ):
        r""" Runs a single training epoch pulled from the dataloader.
        """
        # --- Init Epoch ----
        total_epoch_loss = 0.0
        epoch_batches = self.dataset.dataloader( self.config.neuron.epoch_length )
        progress_bar = qqdm(enumerate(epoch_batches), total=len(epoch_batches), desc=format_str('blue', f'Epoch Progress'))
        for iteration, (inputs) in progress_bar:

            # ---- Forward / Backward ----
            prev_mechanism_weights = self.mechanism_weights.tolist()
            output = self.train ( batch = { 'inputs': inputs } )
            next_mechanism_weights = self.mechanism_weights.tolist()
            total_epoch_loss += output.local_target_loss.item()

            # ---- Logs ----
            self.epoch_logs (
                progress_bar,
                iteration = iteration,
                output = output,
                prev_mechanism_weights = prev_mechanism_weights,
                next_mechanism_weights = next_mechanism_weights
            )
            self.global_step += 1

        self.epoch_loss = total_epoch_loss / self.config.neuron.epoch_length
        self.epoch += 1
        
        # --- wandb save (disabled) ---
        #inputs = torch.rand(10,20)
        #torch.onnx.export(self.nucleus,inputs,'gpt_miner.onnx', opset_version=10)
        #wandb.save('gpt_miner.onnx')

    # ---- Training call ----
    def train ( self, batch: dict ) -> SimpleNamespace:
        r""" Runs a single training batch through the nucleus and applies a gradient update.
            Args:
                batch ( dict, `required`):
                    training batch dictionary.
            Returns:
                output = SimpleNamespace (
                    local_context (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_len, bittensor.__network_dim__)`, `required`):
                        Representations produced by the nucleus's distillation-model prior to producing the hidden units.

                    local_hidden (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_len, bittensor.__network_dim__)`, `required`):
                        Hidden layer representations produced using the local_context.

                    local_target (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_len, bittensor.__vocab_size__)`, `required`):
                        GPT2 MLM target predictions produced using local_hidden.

                    local_target_loss (:obj:`torch.FloatTensor` of shape :obj:`(1)`, `required`):
                        GPT2 MLM loss computed from the local_target.

                    remote_context (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_len, bittensor.__network_dim__)`, `required`):
                        Representations returned from the nucleus.route function after querying the network.

                    remote_hidden (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_len, bittensor.__network_dim__)`, `required`):
                        Hidden layer representations produced using the remote_context.

                    remote_target (:obj:`torch.FloatTensor` of shape :obj:`(batch_size,  bittensor.__vocab_size__)`, `required`):
                        GPT MLM Target predictions produced using remote_hidden.

                    remote_target_loss (:obj:`torch.FloatTensor` of shape :obj:`(1)`, `required`):
                        GPT2 MLM loss computed from the remote_target.

                    distillation_loss (:obj:`torch.FloatTensor` of shape :obj:`(1)`, `required`):
                        Distillation loss between local_context and remote_context.
            )
        """
        # ---- Forward pass ----
        inputs = batch['inputs']
        output = self.nucleus.remote_forward(
            inputs = inputs.to( self.device ),
            training = True,
        )

        # ---- Backward pass ----
        output.loss = output.local_target_loss + output.distillation_loss + output.remote_target_loss
        output.loss.backward() # Accumulates gradients on the nucleus.
        clip_grad_norm_(self.nucleus.parameters(), self.config.neuron.clip_gradients)
        self.optimizer.step() # Applies accumulated gradients.
        self.optimizer.zero_grad() # Zeros out gradients for next accummulation

        # ---- Update global loss ----
        return output

    # ---- Axon Forward call ----
    def forward ( self, pubkey:str, inputs_x: torch.FloatTensor, modality:int ) -> torch.FloatTensor:
        r""" Subscribed to an axon servicing endpoint: processes forward messages from the wire.
            The arguments reflect an RPC request from another miner in the network, the response tensor
            should be the hidden units computed using the local context and with shape: [batch_size, sequence_len, __network_dim__].

            Args:
                pubkey ( str, `required`):
                    The public key of the caller.
                inputs_x ( :obj:`torch.Tensor`, `required`):
                    torch inputs to be forward processed.
                modality ( bittensor.proto.Modality, `required`):
                    modality of inputs e.g. bittensor.proto.Modality.TEXT.

            Returns:
                outputs (:obj:`torch.FloatTensor`):
                    The nucleus's outputs as a torch tensor of shape [batch_size, sequence_len, __network_dim__]
        """
        inputs_x = inputs_x.to( self.device )
        output = self.nucleus.local_forward (
            inputs = inputs_x
        )
        return output.local_hidden

    # ---- Axon Backward call ----
    def backward ( self, pubkey:str, inputs_x:torch.FloatTensor, grads_dy:torch.FloatTensor, modality:int ) -> torch.FloatTensor:
        r""" Subscribed to an axon servicing endpoint: Processes backward messages from the wire.
            Arguments reflect an RPC backward request from another miner in the network, the response tensor
            should be the gradients of the miner's nucleus w.r.t to the inputs_x and the passed output grads_dy.

            Args:
                pubkey ( str, `required`):
                    The public key of the caller.
                inputs_x ( :obj:`torch.Tensor`, `required`):
                    torch inputs from previous forward call.
                grads_dy ( :obj:`torch.Tensor`, `required`):
                    torch grads of forward output.
                modality ( bittensor.proto.Modality, `required`):
                    modality of inputs e.g. bittensor.proto.Modality.TEXT.

            Returns:
                outputs (:obj:`torch.FloatTensor`, `optional`):
                    The gradients w.r.t to the inputs [batch_size, sequence_len, -1]
        """
        if self.config.neuron.compute_remote_gradients:
            with torch.enable_grad():

                # ---- Set up inputs for gradient computations.
                inputs_x.requires_grad = True
                inputs_x = inputs_x.to( self.device )
                grads_dy = grads_dy.to( self.device )
                outputs_y = self.nucleus.local_forward( inputs = inputs_x ).to( self.device )

                # ---- The backward call will accumulate gradients on our parameters.
                if self.config.neuron.accumulate_remote_gradients:
                    torch.autograd.backward (
                        tensors = [outputs_y],
                        grad_tensors = [grads_dy]
                    )
                    return inputs_x.grad if inputs_x.grad != None else None

                # ---- The backward call will simply compute the gradients without accumulating them.
                else:
                    grads_dy = torch.autograd.grad (
                        outputs = outputs_y,
                        inputs = inputs_x,
                        grad_outputs = grads_dy,
                        only_inputs = True,
                        create_graph = False,
                        retain_graph = False
                    )[0]
                    return grads_dy

        # if ! compute_remote_gradients, NO-OP.
        else:
            return None

    def remote ( self, inputs: torch.LongTensor, query: torch.FloatTensor ) -> torch.FloatTensor:
        r""" Subscribed to the nucleus. Called during nucleus.remote_forward. Accepts inputs and
            a query, calls remote endpoints given inputs and query.
            Args:
                inputs (:obj:torch.LongTensor of shape :obj:(batch_size, sequence_dim), required): 
                    Inputs to send on the wire.

                query (:obj:torch.FloatTensor of shape :obj:(batch_size, query_dim), required): 
                    Query tensor used to selected which endpoints to send inputs to.

            Returns:
                response (:obj:torch.FloatTensor of shape :obj:(batch_size, sequence_len, bittensor.__network_dim__), required): 
                    Joined responses from the network call.
        """
        # ---- Topk Weights ----
        topk_weights, topk_uids = self.mechanism_weights.topk( self.config.neuron.sparsity, dim=0)

        # ---- Filter endpoints ----
        endpoints = [self.metagraph.endpoints[uid] for uid in topk_uids]

        # ---- Query network ----
        responses, return_codes = self.dendrite.forward_text( 
            endpoints = endpoints, 
            inputs = [inputs for _ in endpoints] 
        )

        # ---- Join based on weights ----
        joining_weights = F.softmax( topk_weights, dim = 0 )
        output = torch.zeros( (inputs.shape[0], inputs.shape[1], bittensor.__network_dim__)).to(self.device)
        for index, response in enumerate( responses ): 
            output += response.to(self.device) * joining_weights[ index ]
            self.mechanism_weights[topk_uids[index]] = (1 - 0.1) * self.mechanism_weights[topk_uids[index]] + 0.1 * joining_weights[index] # Moving avg update.

        # ---- Return response -----
        return output

    def checkpoint( self ):
        r""" Optionally Saves, updates and then reloads the miner training state.
        """
        last_saved = self.get_saved_state()
        if last_saved == None or last_saved['epoch_loss'] >= self.epoch_loss:
            self.save()
        self.metagraph.load()
        self.metagraph.sync()
        self.metagraph.save()
        self.reload()

    def get_saved_state( self ):
        r""" Returns a saved state dict or none.
        """
        try:
            return torch.load("{}/model.torch".format( self.config.neuron.full_path ))
        except Exception as e:
            logger.exception('Failed to reload model with error: {}', e)
            return None

    def reload( self ):
        r""" Reloads/updates the training state from the disk.
        """
        state_dict = self.get_saved_state()

        # ---- Load training state.
        self.epoch = state_dict['epoch']
        self.epoch_loss = state_dict['epoch_loss']
        self.global_step = state_dict['global_step']

        # ---- Load router and resize to the metagraph size.
        #self.router.sync_with_chain_state( self.metagraph ) # Resize the router.
        #self.router.load_state_dict( state_dict['router_state'], strict=False ) # Load router
        

        # ---- Load nucleus and attach the routing function.
        self.nucleus.load_state_dict( state_dict['nucleus_state'] ) # Load nucleus
        self.nucleus.attach( self )# Re-assign the routing function.

        # --- Load optimizer.
        optim_groups = [
            {"params": self.nucleus.parameters() },
        ]
        self.optimizer = torch.optim.SGD(
            optim_groups,
            lr = state_dict['optimizer_state']['param_groups'][0]['lr'],
            weight_decay = state_dict['optimizer_state']['param_groups'][0]['weight_decay'],
        )

        # ---- Load mechanism weights and pad to size.
        self.mechanism_weights = state_dict['mechanism_weights']
        self.mechanism_weights = torch.nn.functional.pad (
            self.mechanism_weights,
            pad = [0, self.metagraph.n - self.mechanism_weights.numel()],
            value=0
        )
        bittensor.logging.success( prefix = 'Reloaded model', sufix = '<blue>{}/model.torch</blue>'.format( self.config.neuron.full_path ))

    def save( self ):
        r""" Saves the training state to disk.
        """
        try:
            state_dict = {
                'epoch': self.epoch,
                'epoch_loss': self.epoch_loss,
                'global_step': self.global_step,
                'mechanism_weights': self.mechanism_weights, # Save row.
                'nucleus_state': self.nucleus.state_dict(), # Save nucleus state.
                'optimizer_state': self.optimizer.state_dict(), # Save optimizer.
            }
            torch.save( state_dict, "{}/model.torch".format( self.config.neuron.full_path, self.epoch_loss ) )
            bittensor.logging.success(prefix='Saved model', sufix='<blue>{}/model.torch</blue>'.format( self.config.neuron.full_path ) )
        except Exception as e:
             logger.exception('Failed to save model with error:{}', e)

    def set_mechanism_weights( self ):
        r""" Sets the mechanism weights on chain.
        """
        try:
            uids = self.metagraph.uids
            did_set = self.subtensor.set_weights(
                wallet = self.wallet,
                uids = uids,
                weights = self.mechanism_weights,
                wait_for_inclusion = True
            )
            if did_set:
                logger.success('Set weights:'.ljust(20) + '{}', self.mechanism_weights.tolist())
            else:
                logger.warning('Failed to set weights on chain.')
                self.subtensor = bittensor.subtensor( config = self.config.subtensor )
                self.subtensor.connect()

        except Exception as e:
            logger.error('Failure setting weights on chain with error: {}', e)

    def startup( self ):
        r""" Starts and subscribes the miner.
        """
        # ---- Setup UPNPC ----
        if self.config.neuron.use_upnpc:
            bittensor.logging.success(prefix = 'Set upnpc', sufix = '<green>ON</green>')
            try:
                self.external_port = net.upnpc_create_port_map( port = self.axon.port )
            except net.UPNPCException as upnpc_exception:
                logger.critical('Failed to hole-punch with upnpc')
                raise RuntimeError('Failed to hole-punch with upnpc')
        else:
            bittensor.logging.success(prefix = 'Set upnpc', sufix = '<red>OFF</red>')
            self.external_port = self.config.axon.port

        # ---- Get external ip ----
        try:
            self.external_ip = net.get_external_ip()
            bittensor.logging.success(prefix = 'External IP', sufix = '<blue>{}</blue>'.format(self.external_ip))
        except net.ExternalIPNotFound as external_port_exception:
            raise RuntimeError('Unable to attain your external ip. Check your internet connection. error:{}', external_port_exception)

        # ---- Setup tensorboard ----
        if self.config.neuron.use_tensorboard == True:
            self._tensorboard_program = program.TensorBoard()
            self._tensorboard_program.configure(argv=[None, '--logdir', self.config.neuron.full_path, '--load_fast=true'])
            self._tensorbaord_url = self._tensorboard_program.launch()
            bittensor.logging.success(prefix = 'Set tensorboard', sufix = '<blue>http://localhost:6006/</blue>')
        else: bittensor.logging.success(prefix = 'Set tensorboard', sufix = '<red>OFF</red>')

        # ---- Setup Wallet. ----
        if not self.wallet.has_coldkeypub:
            self.wallet.create_new_coldkey( n_words = 12, use_password = True )
        if not self.wallet.has_coldkeypub:
            raise RuntimeError('Miner must have access to a decrypted coldkeypub')
        if not self.wallet.has_hotkey:
            self.wallet.create_new_hotkey( n_words = 12, use_password = False )
        if not self.wallet.has_hotkey:
            raise RuntimeError('Miner must have access to a decrypted hotkey')

        # ---- Subscribe to chain ----
        subscribe_success = self.subtensor.subscribe(
                wallet = self.wallet,
                ip = self.external_ip,
                port = self.external_port,
                modality = bittensor.proto.Modality.TEXT,
                wait_for_finalization = True,
                timeout = 4 * bittensor.__blocktime__,
        )
        if not subscribe_success:
            raise RuntimeError('Failed to subscribe neuron.')

        # ---- Starting axon ----
        self.axon.start()

    def shutdown ( self ):
        r""" Shutsdown the miner and it's dependencies.
        """
        # ---- Stop axon ----
        self.axon.stop()

    # ---- QQDM Training logs ----
    def epoch_logs( self, progress_bar, iteration:int, output: SimpleNamespace, prev_mechanism_weights: List[float], next_mechanism_weights: List[float] ):
        r""" Called after every training step. Displays miner state to screen.
        """
        self_uid = self.metagraph.hotkeys.index( self.wallet.hotkey.public_key )
        stake = self.metagraph.S[ self_uid ].item()
        rank = self.metagraph.R[ self_uid ].item()
        incentive = self.metagraph.I[ self_uid ].item()
        info = {
            'GS': colored('{}'.format(self.global_step), 'red'),
            'LS': colored('{}'.format(iteration), 'blue'),
            'Epoch': colored('{}'.format(self.epoch+1), 'green'),
            'Loss': colored('{:.4f}'.format(self.epoch_loss), 'yellow'),
            'Best': colored('{:.4f}'.format(self.best_epoch_loss), 'red'),
            'L-loss': colored('{:.4f}'.format(output.local_target_loss.item()), 'blue'),
            'R-loss': colored('{:.4f}'.format(output.remote_target_loss.item()), 'green'),
            'D-loss': colored('{:.4f}'.format(output.distillation_loss.item()), 'yellow'),
            'nPeers': colored(self.metagraph.n.item(), 'red'),
            'Stake(\u03C4)': colored('{:.3f}'.format(stake), 'green'),
            'Rank(\u03C4)': colored('{:.3f}'.format(rank), 'blue'),
            'Incentive(\u03C4/block)': colored('{:.6f}'.format(incentive), 'yellow'),
        }
        for uid in self.metagraph.uids.tolist():
            #if uid in self.topk_indices:
                #wandb.log({'Mechanism_weights_' + str(uid):next_mechanism_weights[uid]})
                
            if next_mechanism_weights[uid] != 0:
                weight_dif = next_mechanism_weights[uid] - prev_mechanism_weights[uid]

                if weight_dif > 0:
                    info[colored(str(uid), 'green')] = colored('{:.4f}'.format(next_mechanism_weights[uid]), 'green')
                elif weight_dif == 0:
                    info[str(uid)] = colored('{:.4f}'.format(next_mechanism_weights[uid]), 'white')
                else:
                    info[colored(str(uid), 'red')] = colored('{:.4f}'.format(next_mechanism_weights[uid]), 'red')

        progress_bar.set_infos( info )
        wandb.log({
            'remote_target_loss':output.remote_target_loss.item(),
            'distillation_loss':output.distillation_loss.item(), 
            "local_target_loss": output.local_target_loss.item(),
            'Number of Peers':self.metagraph.n.item(),
            'Stake':stake,
            'Rank':rank,
            'Incentive':incentive}
        )
        if self.config.neuron.use_tensorboard:
            self.tensorboard.add_scalar('R-loss', output.remote_target_loss.item(), self.global_step)
            self.tensorboard.add_scalar('L-loss', output.local_target_loss.item(), self.global_step)
            self.tensorboard.add_scalar('D-loss', output.distillation_loss.item(), self.global_step)


if __name__ == "__main__":
    neuron().run()
