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

""" Mixture-of-experts weighting and distillation learning test for template_miner.
"""

import math
import wandb
import torch
import argparse
import bittensor
import transformers
from pathlib import Path

from datasets import load_dataset
from types import SimpleNamespace
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import GPT2TokenizerFast, GPT2LMHeadModel
from transformers.modeling_utils import Conv1D
from torch.nn import TransformerEncoder, TransformerEncoderLayer

# from bittensor._neuron.text.template_miner.nucleus_impl import Nucleus
from nucleus_344_9_2 import Nucleus
from nucleus_344_9_3 import Nucleus as Nucleus3
from nucleus_344_9_5 import Nucleus as Nucleus5


def modify_args(parser: argparse.ArgumentParser):
    r""" Modify custom params in the parser for this test.
    """
    parser.add_argument('--wandb.name', type=str, help='''Optionally pass wandb run name for use_wandb''',
                        default='BIT-344-adv-12')
    parser.add_argument('--wandb.project', type=str, help='''Optionally pass wandb project name for use_wandb''',
                        default='neuron-tests-adv')
    parser.add_argument('--wandb.tags', type=str, help='''Optionally pass wandb tags for use_wandb''',
                        default='hf losses, no-pos-enc, neuron, test, template_miner_distil, gpt2, '
                                'remotes weighted-join, mixture-of-experts, rescoring')
    parser.add_argument('--wandb.run_group', type=str, help='''Optionally pass wandb group name for use_wandb''',
                        default='template_miner_adv-12')

    parser.add_argument('--dataset.batch_size', type=int, help='Batch size.', default=8)
    parser.add_argument('--dataset.block_size', type=int, help='Number of text items to pull for each example..',
                        default=80)
    parser.add_argument('--dataset.num_workers', type=int, help='Number of workers for data loader.', default=5)
    parser.add_argument('--dataset.name', type=str, help='Which dataset to use.', default='bookcorpusopen')
    parser.add_argument('--dataset.split', type=str, help='Which split to use (train/test/validation).',
                        default='train')

    parser.add_argument('--nucleus.nhid', type=int,
                        help='the dimension of the feedforward network model in nn.TransformerEncoder', default=768)
    parser.add_argument('--nucleus.nhead', type=int, help='the number of heads in the multihead attention models',
                        default=8)
    parser.add_argument('--nucleus.nlayers', type=int,
                        help='the number of nn.TransformerEncoderLayer in nn.TransformerEncoder', default=8)
    parser.add_argument('--nucleus.nlayers_local_hidden', type=int,
                        help='the number of nn.TransformerEncoderLayer in nn.TransformerEncoder', default=2)
    parser.add_argument('--nucleus.nlayers_distil_hidden', type=int,
                        help='the number of nn.TransformerEncoderLayer in nn.TransformerEncoder', default=1)
    parser.add_argument('--nucleus.nlayers_remote_hidden', type=int,
                        help='the number of nn.TransformerEncoderLayer in nn.TransformerEncoder', default=2)
    parser.add_argument('--nucleus.dropout', type=float, help='the dropout value', default=0.1)

    # From: https://github.com/huggingface/transformers/blob/master/examples/research_projects/distillation/train.py
    parser.add_argument(
        "--nucleus.gradient_accumulation_steps",
        type=int,
        default=8,
        help="Gradient accumulation for larger training batches.",
    )
    parser.add_argument("--nucleus.temperature", default=2.0, type=float,
                        help="Temperature for the softmax temperature.")
    parser.add_argument(
        "--nucleus.alpha_ce", default=0.5, type=float, help="Linear weight for the distillation loss. Must be >=0."
    )
    parser.add_argument("--nucleus.alpha_clm", default=0.5, type=float,
                        help="Linear weight for the CLM loss. Must be >=0.")
    parser.add_argument("--nucleus.alpha_clm_dis", default=0.0, type=float,
                        help="Linear weight for the CLM distillation loss. Must be >=0.")
    parser.add_argument("--nucleus.alpha_mse", default=0.0, type=float,
                        help="Linear weight of the MSE loss. Must be >=0.")
    parser.add_argument("--nucleus.alpha_mse_hid", default=0.0, type=float,
                        help="Linear weight of the hidden MSE loss. Must be >=0.")
    parser.add_argument(
        "--nucleus.alpha_cos", default=0.0, type=float, help="Linear weight of the cosine embedding loss. Must be >=0."
    )

    parser.add_argument('--neuron.expert_len', type=int, help='Number of experts.', default=7)

    parser.add_argument('--neuron.learning_rate', type=float, help='Training initial learning rate.', default=7e-5)
    parser.add_argument('--neuron.weight_decay', type=float, help='nucleus parameter weight decay.', default=0.25)
    parser.add_argument('--neuron.momentum', type=float, help='optimizer momentum.', default=0.8)
    parser.add_argument('--neuron.clip_gradients', type=float,
                        help='Implement gradient clipping to avoid exploding loss on smaller architectures.',
                        default=1.0)
    parser.add_argument('--neuron.batch_size_train', type=int, help='Training batch size.', default=32)
    parser.add_argument('--neuron.device', type=str, help='Torch device for training.', default='cuda:0')
    parser.add_argument('--neuron.second_device', type=str, help='Torch second device training.',
                        default='cuda:0')
    parser.add_argument('--neuron.use_wandb', action='store_true',
                        help='''neuron activates its weights and biases powers''', default=False)
    parser.add_argument('--neuron.n_epochs', type=int, help='Number of training epochs.', default=500000)
    parser.add_argument('--neuron.lr_scheduler', type=str, help='Learning rate scheduler name.',
                        default='get_cosine_schedule_with_warmup')
    parser.add_argument('--neuron.num_warmup_steps', type=int, help='Learning rate scheduler number of warmup steps.',
                        default=30000)
    parser.add_argument('--neuron.num_cycles', type=int,
                        help='Learning rate scheduler number of cycles for hard restart.', default=15)
    parser.add_argument('--neuron.learning_rate_chain', type=float, help='Training initial learning rate.', default=1)
    parser.add_argument('--neuron.weight_decay', type=float, help='nucleus parameter weight decay.', default=0.25)
    parser.add_argument('--neuron.momentum', type=float, help='optimizer momentum.', default=0.8)
    parser.add_argument('--neuron.clip_gradients', type=float,
                        help='Implement gradient clipping to avoid exploding loss on smaller architectures.',
                        default=1.0)


def main_config() -> 'bittensor.Config':
    r""" Fills a config namespace object with defaults or information from the command line.
    """
    parser = argparse.ArgumentParser(conflict_handler='resolve')
    parser.add_argument('--config', type=str, help='If set, defaults are overridden by passed file.')

    bittensor.logging.add_args(parser)
    bittensor.wandb.add_args(parser)
    bittensor.dataset.add_args(parser)
    bittensor._neuron.text.template_miner.nucleus.add_args(parser)

    modify_args(parser)

    return bittensor.config(parser)


def chunk(batch, block_size: int):
    r"""
    Concatenates and chunks a batch of token sequences into batches of length block_size.
    Args:
        batch: Input batch of tokenized sequences.
        block_size: Length of each token sequence in the batch.

    Returns:
        A new modified batch of shape [new_batch_size, block_size].
    """
    concatenated = {key: sum(batch[key], []) for key in batch.keys()}
    total_length = len(concatenated['input_ids'])
    trunc_length = (total_length // block_size) * block_size
    new_batch = {
        key: [val[i:i + block_size] for i in range(0, trunc_length, block_size)] for key, val in concatenated.items()
    }
    return new_batch


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class Random(nn.Module):
    def __init__(self, config):
        super(Random, self).__init__()

        self.config = config

        self.embedding = nn.Embedding(bittensor.__vocab_size__, bittensor.__network_dim__)
        self.decoder = nn.Linear(bittensor.__network_dim__, bittensor.__vocab_size__)

        # Local Model
        local_layers = TransformerEncoderLayer(bittensor.__network_dim__, self.config.nucleus.nhead,
                                               self.config.nucleus.nhid, self.config.nucleus.dropout,
                                               activation='gelu')
        self.local_pos_encoder = PositionalEncoding(bittensor.__network_dim__, self.config.nucleus.dropout)
        self.local_encoder = TransformerEncoder(local_layers, self.config.nucleus.nlayers)

        initrange = 0.1
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input_ids: torch.LongTensor, output_hidden_states: bool = False) -> SimpleNamespace:
        output = SimpleNamespace()
        # embedding: retrieve learned representation vectors for input vocabulary tokens.
        # inputs.shape = [batch_size, sequence_len]
        # embedding.shape = [batch_size, sequence_len, bittensor.__network_dim__]
        embedding = self.embedding(input_ids) * math.sqrt(bittensor.__network_dim__)

        # embedding.shape = [batch_size, sequence_len, bittensor.__network_dim__]
        # local_encoder expects embedding.shape = [sequence_len, batch_size, bittensor.__network_dim__]
        embedding = embedding.transpose(0, 1)

        # pos_embedding: adding positional encoding to embedding.
        # pos_embedding.shape = [sequence_len, batch_size, bittensor.__network_dim__]
        pos_embedding = self.local_pos_encoder(embedding)

        # https://pytorch.org/docs/1.8.1/generated/torch.nn.Transformer.html#torch.nn.Transformer.forward
        # src: (S, N, E) the sequence to the encoder (required).
        # src_mask: (S, S) the mask for the src sequence (optional).
        # where S is the source sequence length, N is the batch size, E is the feature number

        # inputs.shape = [batch_size, sequence_len]
        sequence_len = input_ids.shape[1]

        # src_mask: attention mask adds -inf to positions not allowed to attend, preventing forward-looking when
        #           predicting each token in the sequence.
        # src_mask.shape = [sequence_len, sequence_len]
        src_mask = torch.triu(torch.ones(sequence_len, sequence_len) * float('-inf'), diagonal=1)
        src_mask = src_mask.to(self.config.neuron.device)

        # local_context: hidden layer encoding of sequence with local_context.
        # local_context.shape = [sequence_len, batch_size, bittensor.__network_dim__]
        local_context = self.local_encoder(pos_embedding, mask=src_mask)  # base features

        # external expects output.local_context.shape = [batch_size, sequence_len, bittensor.__network_dim__]
        output.local_context = local_context.transpose(0, 1)

        output.logits = self.decoder(output.local_context)

        if output_hidden_states:
            output.hidden_states = [output.local_context]

        return output


def main(config: 'bittensor.Config'):
    r"""
    Trains template_miner nucleus local transformer model on a large dataset, with a next token prediction objective.
    Use as test to evaluate next token prediction accuracy by comparing against pretrained model baseline.
    Use as validation check with expectation of similar train/validation accuracy, to ensure no label leak
    in the training process which would produce much larger train accuracy than validation accuracy.
    Args:
        config (:obj:`bittensor.Config`, `required`): bittensor config

    Returns:

    """
    print("{}{}".format(config.wandb.directory, config.wandb.name))
    Path("{}{}".format(config.wandb.directory, config.wandb.name)).mkdir(exist_ok=True)

    batch_size = config.dataset.batch_size
    block_size = config.dataset.block_size

    # Load a named dataset split from HuggingFace datasets.
    dataset = load_dataset(config.dataset.name, split=config.dataset.split)

    # Tokenize the dataset text sequences.
    # tokenizer = bittensor.tokenizer()
    teacher_model_series = 'gpt2'
    tokenizer = GPT2TokenizerFast.from_pretrained(teacher_model_series, local_files_only=False)
    dataset = dataset.map(lambda _batch: tokenizer(_batch['text']), remove_columns=['text', 'title'],
                          batched=True, num_proc=config.dataset.num_workers)

    # Chunk the token sequences into fixed block_size length.
    dataset = dataset.map(lambda _batch: chunk(_batch, block_size),
                          batched=True, batch_size=2, num_proc=config.dataset.num_workers)  #

    # Format our dataset to outputs torch.Tensor to train a pytorch model.
    columns = ['input_ids', 'attention_mask']
    dataset.set_format(type='torch', columns=columns)

    distil_device = config.neuron.device

    # Choose teacher models with significantly different capacity to observe learning of
    #  significantly different peer weights for these by the nucleus distillation model.
    teachers = [
                {'name': 'random',  # 0: random encoder
                 'dim': 1024,
                 'device': config.neuron.device,
                 'model': Random(config).half().to(config.neuron.device)},
                {'name': 'distilgpt2',  # 1: 6-layer, 768-hidden, 12-heads, 82M parameters
                 'model-name': 'distilgpt2',  # 6-layer, 768-hidden, 12-heads, 82M parameters
                 'dim': 768,
                 'device': config.neuron.device},
                {'name': 'gpt2',  # 2: 12-layer, 768-hidden, 12-heads, 117M parameters.
                 'model-name': 'gpt2',  # 12-layer, 768-hidden, 12-heads, 117M parameters.
                 'dim': 768,
                 'device': config.neuron.device},
                {'name': 'gpt2-medium',  # 3: 24-layer, 1024-hidden, 16-heads, 345M parameters.
                 'model-name': 'gpt2-medium',  # 24-layer, 1024-hidden, 16-heads, 345M parameters.
                 'dim': 1024,
                 'device': config.neuron.device},
                {'name': 'gpt2-large',  # 36-layer, 1280-hidden, 20-heads, 774M parameters.
                 'model-name': 'gpt2-large',  # 36-layer, 1280-hidden, 20-heads, 774M parameters.
                 'dim': 1280,
                 'device': config.neuron.device},
                # {'name': 'gpt2-xl',  # 48-layer, 1600-hidden, 25-heads, 1558M parameters.
                #  'dim': 1600,
                #  'device': config.neuron.second_device},
                {'name': 'gpt2-medium-adv',  # baseline, 25-layer, 1024-hidden, 16-heads, 345M parameters.
                 'dim': 1024,
                 'device': config.neuron.device}
                ]

    for teacher in teachers:
        if 'model-name' in teacher:
            # Load pretrained teacher models with language-modeling heads
            teacher['model'] = GPT2LMHeadModel.from_pretrained(teacher['model-name']).half().to(teacher['device'])

    adaptors = {}
    for dim in [768, 1024, 1280]:
        adaptor = nn.Linear(dim, bittensor.__network_dim__, bias=False)
        adaptors[dim] = adaptor.state_dict()

    transform_layers = TransformerEncoderLayer(bittensor.__network_dim__, config.nucleus.nhead,
                                               config.nucleus.nhid, config.nucleus.dropout,
                                               activation='gelu')
    extend_transform = TransformerEncoder(transform_layers, 1)
    extend_state = extend_transform.state_dict()

    # load_path = None
    load_path = "{}{}".format(config.wandb.directory, 'BIT-344-adv-10')

    for t, teacher in enumerate(teachers):
        # Adapt the teacher hidden dimension to the bittensor network dimension of bittensor by using
        #  a fully-connected layer to convert teacher hidden features to a size of bittensor.__network_dim__
        teacher['adaptor'] = nn.Linear(teacher['dim'], bittensor.__network_dim__, bias=False).to(teacher['device'])
        if load_path is None:
            teacher['adaptor'].load_state_dict(adaptors[teacher['dim']], strict=True)
        else:
            model_state = torch.load('{}/teacher-{}-adaptor.torch'.format(load_path, t))
            teacher['adaptor'].load_state_dict(model_state, strict=True)

        transform_layers = TransformerEncoderLayer(bittensor.__network_dim__, config.nucleus.nhead,
                                                   config.nucleus.nhid, config.nucleus.dropout,
                                                   activation='gelu')
        teacher['extend'] = TransformerEncoder(transform_layers, 1).to(teacher['device'])
        if load_path is None:
            teacher['extend'].load_state_dict(extend_state, strict=True)
        else:
            model_state = torch.load('{}/teacher-{}-extend.torch'.format(load_path, t))
            teacher['extend'].load_state_dict(model_state, strict=True)

    torch.cuda.empty_cache()

    # a single common decoder for all teachers to promote similar magnitudes
    teacher_decoder = nn.Linear(bittensor.__network_dim__, bittensor.__vocab_size__, bias=False).to(config.neuron.device)
    if load_path is None:
        initrange = 0.1
        teacher_decoder.weight.data.uniform_(-initrange, initrange)
    else:
        model_state = torch.load('{}/teacher-decoder.torch'.format(load_path))
        teacher_decoder.load_state_dict(model_state, strict=True)

    # Learn the dimension adaptors for the input teachers getting mixed
    adaptor_params = (sum((list(teacher['adaptor'].parameters()) for teacher in teachers), []) +
                      sum((list(teacher['extend'].parameters()) for teacher in teachers), []) +
                      list(teacher_decoder.parameters()))
    adaptor_optimizer = torch.optim.AdamW(adaptor_params, lr=config.neuron.learning_rate)

    # src_mask: attention mask adds -inf to positions not allowed to attend, preventing forward-looking when
    #           predicting each next token in the sequence.
    # src_mask.shape = [sequence_len, sequence_len]
    sequence_len = block_size - 1
    src_mask = torch.triu(torch.ones(sequence_len, sequence_len) * float('-inf'), diagonal=1)
    src_mask = src_mask.to(config.neuron.device)

    torch.cuda.empty_cache()

    # Initialize nucleus pytorch model to perform distillation from teacher and move to specified device
    distil_config = config.copy()
    distil_config.neuron.device = distil_device
    distil_config.nucleus.alpha_clm = 1.
    distil_config.nucleus.alpha_clm_dis = 0.
    distil_config.nucleus.alpha_clm_rmt = 1.
    distil_config.nucleus.alpha_mse = 0.0
    distil_config.nucleus.alpha_mse_hid = 1.
    distil_config.nucleus.alpha_ce = 0.
    distil_config.nucleus.alpha_cos = 0.
    distil_model = Nucleus(distil_config).to(distil_device)
    # Accommodate multiple remote teachers for this experiment
    distil_model.peer_weights = nn.Parameter(torch.ones([len(teachers)], requires_grad=True, device=distil_device))
    # Save model to capture unique parameter initialization for reuse in other distil model.
    distil_state = distil_model.state_dict()

    print('distil', distil_model.alpha_ce, distil_model.alpha_clm, distil_model.alpha_clm_dis,
          distil_model.alpha_clm_rmt, distil_model.alpha_mse, distil_model.alpha_mse_hid, distil_model.alpha_cos)

    # Initialize another nucleus that distils from an annealed sgmoe
    predistil_device = config.neuron.device
    predistil_config = config.copy()
    predistil_config.neuron.device = predistil_device
    predistil_config.nucleus.alpha_clm = 1.
    predistil_config.nucleus.alpha_clm_dis = 0.0
    predistil_config.nucleus.alpha_clm_rmt = 1.0
    predistil_config.nucleus.alpha_mse = 0.0
    predistil_config.nucleus.alpha_mse_hid = 1.
    predistil_config.nucleus.alpha_ce = 0.0
    predistil_config.nucleus.alpha_cos = 0.0
    predistil_model = Nucleus3(predistil_config)
    predistil_model.peer_weights = nn.Parameter(
        torch.ones([len(teachers)], requires_grad=True, device=predistil_device))
    # Load same initialization as distil_model
    predistil_model.load_state_dict(distil_state, strict=True)
    predistil_model = predistil_model.to(predistil_device)

    print(predistil_model)
    print('predistil', predistil_model.alpha_ce, predistil_model.alpha_clm, predistil_model.alpha_clm_dis,
          predistil_model.alpha_clm_rmt, predistil_model.alpha_mse, predistil_model.alpha_mse_hid,
          predistil_model.alpha_cos)

    # Initialize another nucleus that distils from an annealed sgmoe
    advdistil_device = config.neuron.device
    advdistil_config = config.copy()
    advdistil_config.neuron.device = advdistil_device
    advdistil_config.nucleus.alpha_clm = 1.
    advdistil_config.nucleus.alpha_clm_dis = 0.0
    advdistil_config.nucleus.alpha_clm_rmt = 1.0
    advdistil_config.nucleus.alpha_mse = 0.0
    advdistil_config.nucleus.alpha_mse_hid = 1.
    advdistil_config.nucleus.alpha_ce = 0.0
    advdistil_config.nucleus.alpha_cos = 0.0
    advdistil_model = Nucleus5(advdistil_config)
    # Load same initialization as distil_model
    advdistil_model.load_state_dict(distil_state, strict=False)
    if load_path is not None:
        advdistil_model.load('{}/advdistil_'.format(load_path))
    advdistil_model = advdistil_model.to(advdistil_device)

    print(advdistil_model)
    print('advdistil', advdistil_model.alpha_ce, advdistil_model.alpha_clm, advdistil_model.alpha_clm_dis,
          advdistil_model.alpha_clm_rmt, advdistil_model.alpha_mse, advdistil_model.alpha_mse_hid,
          advdistil_model.alpha_cos)

    # Initialize another nucleus that learns an lm head but without distillation
    undistil_device = config.neuron.device
    undistil_config = config.copy()
    undistil_config.neuron.device = undistil_device
    undistil_config.nucleus.alpha_clm = 1.
    undistil_config.nucleus.alpha_clm_dis = 0.0
    undistil_config.nucleus.alpha_clm_rmt = 0.0
    undistil_config.nucleus.alpha_mse = 0.0
    undistil_config.nucleus.alpha_mse_hid = 0.0
    undistil_config.nucleus.alpha_ce = 0.0
    undistil_config.nucleus.alpha_cos = 0.0
    undistil_model = Nucleus(undistil_config)
    # undistil model won't distil, but need to create same-size parameter to load same initialization
    undistil_model.peer_weights = nn.Parameter(torch.ones([len(teachers)], requires_grad=True, device=distil_device))
    # Load same initialization as distil_model
    undistil_model.load_state_dict(distil_state, strict=True)
    undistil_model = undistil_model.to(undistil_device)

    print(undistil_model)
    print('undistil', undistil_model.alpha_ce, undistil_model.alpha_clm, undistil_model.alpha_clm_dis,
          undistil_model.alpha_clm_rmt, undistil_model.alpha_mse, undistil_model.alpha_mse_hid,
          undistil_model.alpha_cos)

    # Original optimizer in template-miner, but learning rate of 1 is too high for this scenario since the adaptors
    #  first need to get trained before teacher capabilities can be discerned.
    # So we opt for using the AdamW with lower learning rate also for the peer weight learning.

    # distil_weight_optimizer = torch.optim.SGD(
    #     [{'params': distil_model.peer_weights,
    #       'lr': distil_model.config.neuron.learning_rate_chain,
    #       'momentum': distil_model.config.neuron.momentum}]
    # )
    # distil_weight_scheduler = torch.optim.lr_scheduler.StepLR(distil_weight_optimizer, step_size=1000, gamma=0.995)

    # print(len(list(distil_model.parameters())), len(list(filter(lambda p: id(p) != id(distil_model.peer_weights), distil_model.parameters()))))
    # Define optimizer over all model parameters at specified learning rate
    # distil_optimizer = torch.optim.AdamW(filter(lambda p: id(p) != id(distil_model.peer_weights), distil_model.parameters()),
    #                                      lr=config.neuron.learning_rate)
    distil_optimizer = torch.optim.AdamW(distil_model.parameters(), lr=config.neuron.learning_rate)
    predistil_optimizer = torch.optim.AdamW(predistil_model.parameters(), lr=config.neuron.learning_rate)
    advdistil_optimizer = torch.optim.AdamW(advdistil_model.parameters(), lr=config.neuron.learning_rate)
    undistil_optimizer = torch.optim.AdamW(undistil_model.parameters(), lr=config.neuron.learning_rate)

    # Define learning rate scheduler (multiplier) for optimizer
    distil_scheduler = None
    predistil_scheduler = None
    undistil_scheduler = None
    adaptor_scheduler = None

    if config.neuron.lr_scheduler == 'get_cosine_schedule_with_warmup':
        adaptor_scheduler = transformers.get_cosine_schedule_with_warmup(optimizer=adaptor_optimizer,
                                                                         num_warmup_steps=config.neuron.num_warmup_steps,
                                                                         num_training_steps=config.neuron.n_epochs)
        distil_scheduler = transformers.get_cosine_schedule_with_warmup(optimizer=distil_optimizer,
                                                                        num_warmup_steps=config.neuron.num_warmup_steps,
                                                                        num_training_steps=config.neuron.n_epochs)
        predistil_scheduler = transformers.get_cosine_schedule_with_warmup(optimizer=predistil_optimizer,
                                                                           num_warmup_steps=config.neuron.num_warmup_steps,
                                                                           num_training_steps=config.neuron.n_epochs)
        advdistil_scheduler = transformers.get_cosine_schedule_with_warmup(optimizer=advdistil_optimizer,
                                                                           num_warmup_steps=config.neuron.num_warmup_steps,
                                                                           num_training_steps=config.neuron.n_epochs)
        undistil_scheduler = transformers.get_cosine_schedule_with_warmup(optimizer=undistil_optimizer,
                                                                          num_warmup_steps=config.neuron.num_warmup_steps,
                                                                          num_training_steps=config.neuron.n_epochs)

    if config.neuron.use_wandb:
        bittensor.wandb(config)  # Initialize wandb logging
        # wandb.watch(distil_model)  # Track model parameters and gradients
        # wandb.watch(predistil_model)  # Track model parameters and gradients
        wandb.watch(advdistil_model)  # Track model parameters and gradients
        # wandb.watch(undistil_model)  # Track model parameters and gradients
        wandb.watch(teachers[-1]['extend'])  # Track model parameters and gradients
        wandb_table_data = []

    torch.cuda.empty_cache()

    for dataset_epoch in range(1):
        # Define pytorch dataloader with shuffled batches of batch_size token sequences of block_size length.
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch, batch in enumerate(dataloader):
            input_ids = batch['input_ids'].to(distil_device)
            target = input_ids[:, -1]  # held out target of last token
            input_ids = input_ids[:, :-1]  # entire sequence except last token

            teacher_inputs = {}
            teacher_losses = 0
            for teacher in teachers[:-1]:
                with torch.no_grad():
                    if teacher['device'] not in teacher_inputs:
                        teacher_inputs[teacher['device']] = input_ids.clone().to(teacher['device'])

                    teacher_input_ids = teacher_inputs[teacher['device']]
                    teacher_output = teacher['model'](input_ids=teacher_input_ids, output_hidden_states=True)
                    teacher['hidden_states'] = teacher_output.hidden_states[-1].float().detach()
                    # teacher['hidden_states'].shape = [16, 79, 1024]

                if teacher['name'] == 'gpt2-medium':
                    teacher_ext = teachers[-1]
                    with torch.no_grad():
                        hidden_states = teacher['hidden_states'].clone().to(teacher_ext['device'])

                    teacher_ext['hidden_states'] = teacher_ext['adaptor'](hidden_states).transpose(0, 1)
                    # hidden_states.shape = [79, 16, 1024]
                    # no masking: baseline
                    teacher_ext['hidden_states'] = teacher_ext['extend'](teacher_ext['hidden_states']).transpose(0, 1)

                    # local_target: projection of local_hidden onto target dimension.
                    # local_target.shape = [batch_size, sequence_len, bittensor.__vocab_size__]
                    local_target = teacher_decoder(teacher_ext['hidden_states'])

                    # local_target_loss: MLM loss between local_target and passed targets.
                    # local_target_loss.shape = [1]
                    shift_logits = local_target[..., :-1, :].contiguous()
                    shift_labels = teacher_input_ids[..., 1:].contiguous()
                    teacher_ext['loss_clm'] = nn.CrossEntropyLoss()(shift_logits.view(-1, shift_logits.size(-1)),
                                                                    shift_labels.view(-1))

                    teacher_losses += teacher_ext['loss_clm']

                    predictions = shift_logits.detach().max(2).indices
                    teacher_ext['acc'] = (predictions == shift_labels).sum().item() / predictions.nelement()

                    teacher_target = target.clone().to(teacher_ext['device'])
                    teacher_ext['prediction'] = local_target[:, -1, :].argmax(-1)  # predict unseen last token
                    teacher_ext['target_acc'] = (teacher_ext['prediction'] == teacher_target).sum().item() / len(
                        teacher_target)

                teacher['hidden_states'] = teacher['adaptor'](teacher['hidden_states']).transpose(0, 1)
                teacher['hidden_states'] = teacher['extend'](teacher['hidden_states'], mask=src_mask).transpose(0, 1)

                # adapted model task-head performance

                # local_target: projection of local_hidden onto target dimension.
                # local_target.shape = [batch_size, sequence_len, bittensor.__vocab_size__]
                local_target = teacher_decoder(teacher['hidden_states'])

                # local_target_loss: MLM loss between local_target and passed targets.
                # local_target_loss.shape = [1]
                shift_logits = local_target[..., :-1, :].contiguous()
                shift_labels = teacher_input_ids[..., 1:].contiguous()
                teacher['loss_clm'] = nn.CrossEntropyLoss()(shift_logits.view(-1, shift_logits.size(-1)),
                                                            shift_labels.view(-1))

                teacher_losses += teacher['loss_clm']

                predictions = shift_logits.detach().max(2).indices
                teacher['acc'] = (predictions == shift_labels).sum().item() / predictions.nelement()

                teacher_target = target.clone().to(teacher['device'])
                teacher['prediction'] = local_target[:, -1, :].argmax(-1)  # predict unseen last token
                teacher['target_acc'] = (teacher['prediction'] == teacher_target).sum().item() / len(teacher_target)

                with torch.no_grad():
                    # direct model task-head performance
                    shift_logits = teacher_output.logits[..., :-1, :].float().contiguous()
                    shift_labels = teacher_input_ids[..., 1:].contiguous()
                    teacher['dir_loss_clm'] = nn.CrossEntropyLoss()(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                    predictions = shift_logits.detach().max(2).indices
                    teacher['dir_acc'] = (predictions == shift_labels).sum().item() / predictions.nelement()

                    teacher['dir_prediction'] = teacher_output.logits[:, -1, :].argmax(-1)  # predict unseen last token
                    teacher['dir_predictions'] = tokenizer.decode(teacher_output.logits[0].argmax(-1).detach())

                    teacher_target = target.clone().to(teacher['device'])
                    teacher['dir_target_acc'] = (teacher['dir_prediction'] == teacher_target).sum().item() / len(teacher_target)

                    adaptor_lr = adaptor_optimizer.param_groups[0]['lr']

            # Weighted joining of teachers with weights that also get learned
            joining_weights = F.softmax(distil_model.peer_weights, dim=0)
            distil_model.peer_weights_softmax = joining_weights.detach()
            distil_teacher_inputs = None
            for i, teacher in enumerate(teachers):
                if distil_teacher_inputs is None:
                    distil_teacher_inputs = joining_weights[i] * teacher['hidden_states'].detach().to(distil_device)
                else:
                    distil_teacher_inputs += joining_weights[i] * teacher['hidden_states'].detach().to(distil_device)

            distil_output = distil_model.remote_forward(input_ids, training=True,
                                                        teacher_inputs=distil_teacher_inputs)  # forward pass in local transformer model
            distil_total_loss = (distil_model.alpha_clm * distil_output.loss_clm +
                                 distil_model.alpha_clm_dis * distil_output.loss_clm_dis +
                                 distil_model.alpha_clm_rmt * distil_output.loss_clm_rmt +
                                 distil_model.alpha_mse * distil_output.loss_mse +
                                 distil_model.alpha_mse_hid * distil_output.loss_mse_hid +
                                 distil_model.alpha_ce * distil_output.loss_ce +
                                 distil_model.alpha_cos * distil_output.loss_cos)

            with torch.no_grad():
                distil_loss_clm = distil_output.loss_clm
                distil_loss_clm_dis = distil_output.loss_clm_dis
                distil_loss_clm_rmt = distil_output.loss_clm_rmt
                distil_loss_mse = distil_output.loss_mse
                distil_loss_mse_hid = distil_output.loss_mse_hid
                distil_loss_ce = distil_output.loss_ce
                distil_loss_cos = distil_output.loss_cos
                distil_acc = distil_output.local_accuracy  # training accuracy on next token prediction in train sequence with masking
                distil_remote_acc = distil_output.remote_accuracy  # training accuracy on next token prediction in train sequence with masking
                distil_lr = distil_optimizer.param_groups[0]['lr']  # record actual learning rate
                # distil_weight_lr = distil_weight_optimizer.param_groups[0]['lr']  # record actual learning rate

                distil_prediction = distil_output.local_target[:, -1, :].argmax(-1)  # predict unseen last token
                distil_target_acc = (distil_prediction == target).sum().item() / len(
                    target)  # validation accuracy on predicting unseen token

                distil_remote_prediction = distil_output.remote_target[:, -1, :].argmax(-1)  # predict unseen last token
                distil_remote_target_acc = (distil_remote_prediction == target).sum().item() / len(
                    target)  # validation accuracy on predicting unseen token

                predistil_input_ids = input_ids.detach().to(predistil_device)

            # Weighted joining of teachers with weights that also get learned
            joining_weights = F.softmax(predistil_model.peer_weights, dim=0)
            predistil_model.peer_weights_softmax = joining_weights.detach()
            predistil_teacher_inputs = []
            for i, teacher in enumerate(teachers):
                predistil_teacher_inputs += [teacher['hidden_states'].detach().to(predistil_device)]

            # [expert_len * batch_size, sequence_len, bittensor.__network_dim__]
            predistil_teacher_inputs = torch.cat(predistil_teacher_inputs)

            predistil_output = predistil_model.remote_forward(predistil_input_ids, training=True,
                                                              teacher_inputs=predistil_teacher_inputs,
                                                              join_weights=joining_weights)  # forward pass in local transformer model
            predistil_total_loss = (predistil_model.alpha_clm * predistil_output.loss_clm +
                                    predistil_model.alpha_clm_dis * predistil_output.loss_clm_dis +
                                    predistil_model.alpha_clm_rmt * predistil_output.loss_clm_rmt +
                                    predistil_model.alpha_mse * predistil_output.loss_mse +
                                    predistil_model.alpha_mse_hid * predistil_output.loss_mse_hid +
                                    predistil_model.alpha_ce * predistil_output.loss_ce +
                                    predistil_model.alpha_cos * predistil_output.loss_cos)

            with torch.no_grad():
                predistil_loss_clm = predistil_output.loss_clm
                predistil_loss_clm_dis = predistil_output.loss_clm_dis
                predistil_loss_clm_rmt = predistil_output.loss_clm_rmt
                predistil_loss_mse = predistil_output.loss_mse
                predistil_loss_mse_hid = predistil_output.loss_mse_hid
                predistil_loss_ce = predistil_output.loss_ce
                predistil_loss_cos = predistil_output.loss_cos
                predistil_acc = predistil_output.local_accuracy  # training accuracy on next token prediction in train sequence with masking
                predistil_remote_acc = predistil_output.remote_accuracy  # training accuracy on next token prediction in train sequence with masking
                predistil_lr = predistil_optimizer.param_groups[0]['lr']  # record actual learning rate
                # predistil_weight_lr = predistil_weight_optimizer.param_groups[0]['lr']  # record actual learning rate

                predistil_target = target.to(predistil_device)
                predistil_prediction = predistil_output.local_target[:, -1, :].argmax(-1)  # predict unseen last token
                predistil_target_acc = (predistil_prediction == predistil_target).sum().item() / len(
                    predistil_target)  # validation accuracy on predicting unseen token

                # batch_weights.shape = [expert_len * batch_size]
                batch_weights = joining_weights.repeat_interleave(batch_size)

                # predistil_remote_prediction.shape = [expert_len * batch_weights]
                predistil_remote_prediction = predistil_output.remote_target[:, -1, :].argmax(
                    -1)  # predict unseen last token
                predistil_target_expert = predistil_target.repeat(len(teachers))
                predistil_remote_target_acc = (batch_weights * (
                        predistil_remote_prediction == predistil_target_expert).float()).sum().item() / len(
                    predistil_target)  # validation accuracy on predicting unseen token

                advdistil_input_ids = input_ids.detach().to(advdistil_device)

            wandb_adv = {}

            advdistil_output = advdistil_model.remote_forward(advdistil_input_ids, training=True)  # forward pass in local transformer model
            advdistil_target = target.to(advdistil_device)

            advdistil_prediction = advdistil_output.local_target[:, -1, :].argmax(-1)  # predict unseen last token
            wandb_adv['advdistil_target_acc'] = (advdistil_prediction == advdistil_target).sum().item() / len(
                advdistil_target)  # validation accuracy on predicting unseen token

            wandb_adv['advdistil_loss_clm'] = advdistil_output.loss_clm
            wandb_adv['advdistil_acc'] = advdistil_output.local_accuracy  # training accuracy on next token prediction in train sequence with masking
            wandb_adv['advdistil_lr'] = advdistil_optimizer.param_groups[0]['lr']  # record actual learning rate

            advdistil_total_loss = advdistil_model.alpha_clm * advdistil_output.loss_clm  # learn base features

            adv_weights = []
            advdistil_distil_loss_clm_rmt = []
            for e in range(len(teachers)):
                # learn task-head here
                # baseline detection capability
                wandb_adv['advdistil_distil_loss_clm_rmt_%d' % e] = advdistil_output.loss_clm_rmt[e]
                advdistil_distil_loss_clm_rmt += [advdistil_output.loss_clm_rmt[e]]

                wandb_adv['advdistil_distil_acc_%d' % e] = advdistil_output.remote_accuracy[e]  # training accuracy on next token prediction in train sequence with masking
                advdistil_total_loss += advdistil_model.alpha_clm_rmt * advdistil_output.loss_clm_rmt[e]
                adv_weights += [advdistil_output.loss_clm_rmt[e]]

                with torch.no_grad():
                    # advdistil_remote_prediction.shape = [batch_weights]
                    advdistil_remote_prediction = advdistil_output.remote_target[e][:, -1, :].argmax(
                        -1)  # predict unseen last token
                    wandb_adv['advdistil_distil_target_acc_%d' % e] = (advdistil_remote_prediction == advdistil_target).sum().item() / len(
                        advdistil_target)  # validation accuracy on predicting unseen token

            advdistil_model.peer_weights = -torch.tensor(adv_weights)
            advdistil_model.peer_weights_softmax = F.softmax(advdistil_model.peer_weights, dim=0)

            # don't use: Weighted joining of teachers with weights that also get learned
            # joining_weights = F.softmax(advdistil_model.peer_weights, dim=0)
            advdistil_teacher_inputs = []
            for i, teacher in enumerate(teachers):
                advdistil_teacher_inputs += [teacher['hidden_states'].detach().to(advdistil_device)]

            # [expert_len * batch_size, sequence_len, bittensor.__network_dim__]
            advdistil_teacher_inputs = torch.cat(advdistil_teacher_inputs)
            advdistil_output = advdistil_model.remote_forward(advdistil_input_ids, training=True,
                                                              teacher_inputs=advdistil_teacher_inputs)  # forward pass in local transformer model

            advdistil_loss_clm_rmt = []
            for e in range(len(teachers)):
                # learn distil here
                advdistil_total_loss += advdistil_model.alpha_mse_hid * advdistil_output.loss_mse_hid[e]
                wandb_adv['advdistil_loss_mse_hid_%d' % e] = advdistil_output.loss_mse_hid[e]

                # more reliable expert ranking
                wandb_adv['advdistil_loss_clm_rmt_%d' % e] = advdistil_output.loss_clm_rmt[e]
                advdistil_total_loss += advdistil_model.alpha_clm_rmt * advdistil_output.loss_clm_rmt[e]
                advdistil_loss_clm_rmt += [advdistil_output.loss_clm_rmt[e]]
                wandb_adv['advdistil_remote_acc_%d' % e] = advdistil_output.remote_accuracy[e]  # training accuracy on next token prediction in train sequence with masking

                # advdistil_remote_prediction.shape = [batch_weights]
                advdistil_remote_prediction = advdistil_output.remote_target[e][:, -1, :].argmax(
                    -1)  # predict unseen last token
                wandb_adv['advdistil_remote_target_acc_%d' % e] = (advdistil_remote_prediction == advdistil_target).sum().item() / len(
                    advdistil_target)  # validation accuracy on predicting unseen token

            wandb_adv['advdistil_total_loss'] = advdistil_total_loss

            with torch.no_grad():
                advdistil_distil_loss_clm_rmt = torch.stack(advdistil_distil_loss_clm_rmt)
                advdistil_loss_clm_rmt = torch.stack(advdistil_loss_clm_rmt)

                loss_vals = advdistil_distil_loss_clm_rmt
                loss_mean = loss_vals.mean()
                loss_std = loss_vals.std()
                loss_sum = loss_vals.sum()
                loss_excl_mean = 0 * advdistil_distil_loss_clm_rmt
                loss_excl_std = 0 * advdistil_distil_loss_clm_rmt
                n = len(loss_vals)
                for e in range(len(teachers)):
                    wandb_adv['advdistil_distil_loss_clm_rmt_z_%d' % e] = (loss_vals[e] - loss_mean) / loss_std
                    loss_excl_mean[e] = (loss_sum - loss_vals[e]) / (n-1)
                    sqr_dev = (loss_vals - loss_excl_mean[e]) ** 2
                    loss_excl_std[e] = torch.sqrt((sqr_dev.sum() - sqr_dev[e]) / (n-1))  # biased std. dev. estimate
                    wandb_adv['advdistil_distil_loss_clm_rmt_excl_z_%d' % e] = (loss_vals[e] - loss_excl_mean[e]) / loss_excl_std[e]

                loss_vals = advdistil_loss_clm_rmt
                loss_mean = loss_vals.mean()
                loss_std = loss_vals.std()
                loss_sum = loss_vals.sum()
                loss_excl_mean = 0 * advdistil_loss_clm_rmt
                loss_excl_std = 0 * advdistil_loss_clm_rmt
                n = len(loss_vals)
                for e in range(len(teachers)):
                    wandb_adv['advdistil_loss_clm_rmt_z_%d' % e] = (loss_vals[e] - loss_mean) / loss_std
                    loss_excl_mean[e] = (loss_sum - loss_vals[e]) / (n - 1)
                    sqr_dev = (loss_vals - loss_excl_mean[e]) ** 2
                    loss_excl_std[e] = torch.sqrt((sqr_dev.sum() - sqr_dev[e]) / (n - 1))  # biased std. dev. estimate
                    wandb_adv['advdistil_loss_clm_rmt_excl_z_%d' % e] = (loss_vals[e] - loss_excl_mean[e]) / loss_excl_std[e]

                adv_scores = []
                adv_excl_scores = []
                for e in range(len(teachers)):
                    wandb_adv['advdistil_adv_score_%d' % e] = (wandb_adv['advdistil_distil_loss_clm_rmt_z_%d' % e] -
                                                               wandb_adv['advdistil_loss_clm_rmt_z_%d' % e])
                    adv_scores += [wandb_adv['advdistil_adv_score_%d' % e]]

                    wandb_adv['advdistil_adv_excl_score_%d' % e] = (wandb_adv['advdistil_distil_loss_clm_rmt_excl_z_%d' % e] -
                                                               wandb_adv['advdistil_loss_clm_rmt_excl_z_%d' % e])
                    adv_excl_scores += [wandb_adv['advdistil_adv_excl_score_%d' % e]]

                adv_scores = torch.stack(adv_scores)
                adv_loc = adv_scores > 0.  # baseline locations
                advdistil_loss_clm_rmt_u = advdistil_loss_clm_rmt.clone()  # updated
                advdistil_loss_clm_rmt_u_std = advdistil_loss_clm_rmt_u.std()
                advdistil_loss_clm_rmt_u[adv_loc] += adv_scores[adv_loc] * advdistil_loss_clm_rmt_u_std
                advdistil_loss_clm_rmt_u_sm = F.softmax(-advdistil_loss_clm_rmt_u, dim=0)
                advdistil_loss_clm_rmt_sm = F.softmax(-advdistil_loss_clm_rmt, dim=0)

                for e in range(len(teachers)):
                    wandb_adv['advdistil_loss_clm_rmt_u_%d' % e] = advdistil_loss_clm_rmt_u[e]
                    wandb_adv['advdistil_loss_clm_rmt_u_sm_%d' % e] = advdistil_loss_clm_rmt_u_sm[e]
                    wandb_adv['advdistil_loss_clm_rmt_sm_%d' % e] = advdistil_loss_clm_rmt_sm[e]

                adv_excl_scores = torch.stack(adv_excl_scores)
                adv_loc = adv_excl_scores > 0.  # baseline locations
                advdistil_loss_clm_rmt_excl_u = advdistil_loss_clm_rmt.clone()  # updated
                advdistil_loss_clm_rmt_excl_u[adv_loc] += adv_excl_scores[adv_loc] * loss_excl_std[adv_loc]
                advdistil_loss_clm_rmt_excl_u_sm = F.softmax(-advdistil_loss_clm_rmt_excl_u, dim=0)

                for e in range(len(teachers)):
                    wandb_adv['advdistil_loss_clm_rmt_excl_u_%d' % e] = advdistil_loss_clm_rmt_excl_u[e]
                    wandb_adv['advdistil_loss_clm_rmt_excl_u_sm_%d' % e] = advdistil_loss_clm_rmt_excl_u_sm[e]

            torch.cuda.empty_cache()

            undistil_input_ids = input_ids.detach().to(undistil_device)
            undistil_output = undistil_model.local_forward(undistil_input_ids,
                                                           training=True)  # forward pass in local transformer model
            undistil_loss = undistil_output.loss_clm

            with torch.no_grad():
                undistil_acc = undistil_output.local_accuracy  # training accuracy on next token prediction in train sequence with masking
                undistil_lr = undistil_optimizer.param_groups[0]['lr']  # record actual learning rate
                undistil_prediction = undistil_output.local_target[:, -1, :].argmax(-1)  # predict unseen last token
                undistil_target = target.to(undistil_device)
                undistil_target_acc = (undistil_prediction == undistil_target).sum().item() / len(
                    undistil_target)  # validation accuracy on predicting unseen token

                if epoch % 100 == 0:
                    print('%d: %.1f %.1f %.1f '
                          '(%.2f, %.2f, %.2f, '
                          '%.2f, %.2f, %f)' % (epoch, distil_total_loss.item(),
                                               predistil_total_loss.item(),
                                               distil_acc, predistil_acc,
                                               distil_target_acc, distil_remote_target_acc,
                                               teachers[-3]['target_acc'], predistil_target_acc,
                                               distil_lr), end=' ')

                if epoch % 1000 == 0:
                    input_decoded = tokenizer.decode(input_ids[0])
                    distil_predictions = distil_output.local_target[0].detach().argmax(-1)
                    predistil_predictions = predistil_output.local_target[0].detach().argmax(-1)

                    print('\n.\n', input_decoded, '\n...\n')
                    print(list(zip([tokenizer.decode(_) for _ in input_ids[0]],
                                   [tokenizer.decode(_) for _ in distil_predictions])), '\n.\n')

                    distil_predictions = tokenizer.decode(distil_predictions)
                    predistil_predictions = tokenizer.decode(predistil_predictions)
                    if config.neuron.use_wandb:
                        wandb_table_data += [[epoch,
                                              distil_target_acc,
                                              distil_predictions, predistil_predictions, input_decoded] +
                                             [teacher['dir_predictions'] for teacher in teachers[:-1]]]

                if config.neuron.use_wandb:
                    if epoch % 5000 == 0:
                        wandb_table = wandb.Table(columns=['epoch',
                                                           'distil_target_acc',
                                                           'distil_predictions', 'predistil_predictions', 'input'] +
                                                          ['%s' % teacher['name'] for teacher in teachers[:-1]])
                        for row in wandb_table_data:
                            wandb_table.add_data(*row)

                        torch.save(distil_model.state_dict(), "{}{}/distil_model.torch".format(config.wandb.directory, config.wandb.name))
                        advdistil_model.save("{}{}/advdistil_".format(config.wandb.directory, config.wandb.name))

                        torch.save(teacher_decoder.state_dict(),
                                   "{}{}/teacher-decoder.torch".format(config.wandb.directory, config.wandb.name))
                        for t, teacher in enumerate(teachers):
                            torch.save(teacher['adaptor'].state_dict(),
                                       "{}{}/teacher-{}-adaptor.torch".format(config.wandb.directory, config.wandb.name, t))
                            torch.save(teacher['extend'].state_dict(),
                                       "{}{}/teacher-{}-extend.torch".format(config.wandb.directory, config.wandb.name, t))

                    wandb_log = {'distil_loss_clm': distil_loss_clm.item(),
                                 'distil_loss_clm_dis': distil_loss_clm_dis.item(),
                                 'distil_loss_clm_rmt': distil_loss_clm_rmt.item(),
                                 'distil_loss_mse': distil_loss_mse.item(),
                                 'distil_loss_mse_hid': distil_loss_mse_hid.item(),
                                 'distil_loss_ce': distil_loss_ce.item(),
                                 'distil_loss_cos': distil_loss_cos.item(),
                                 'distil_total_loss': distil_total_loss.item(),
                                 'distil_acc': distil_acc,
                                 'distil_remote_acc': distil_remote_acc,
                                 'distil_target_acc': distil_target_acc,
                                 'distil_remote_target_acc': distil_remote_target_acc,
                                 'distil_lr': distil_lr,
                                 'distil_expert_calls': len(teachers) * batch_size * epoch,

                                 'predistil_loss_clm': predistil_loss_clm.item(),
                                 'predistil_loss_clm_dis': predistil_loss_clm_dis.item(),
                                 'predistil_loss_clm_rmt': predistil_loss_clm_rmt.item(),
                                 'predistil_loss_mse': predistil_loss_mse.item(),
                                 'predistil_loss_mse_hid': predistil_loss_mse_hid.item(),
                                 'predistil_loss_ce': predistil_loss_ce.item(),
                                 'predistil_loss_cos': predistil_loss_cos.item(),
                                 'predistil_total_loss': predistil_total_loss.item(),
                                 'predistil_acc': predistil_acc,
                                 'predistil_remote_acc': predistil_remote_acc,
                                 'predistil_target_acc': predistil_target_acc,
                                 'predistil_remote_target_acc': predistil_remote_target_acc,
                                 'predistil_lr': predistil_lr,
                                 'predistil_expert_calls': len(teachers) * batch_size * epoch,

                                 'undistil_loss': undistil_loss.item(),
                                 'undistil_acc': undistil_acc,
                                 'undistil_target_acc': undistil_target_acc,
                                 'undistil_lr': undistil_lr,

                                 'adaptor_lr': adaptor_lr,
                                 'batch': epoch}

                    wandb_log = [wandb_log, wandb_adv]
                    wandb_log = wandb_log + [{'teacher%d_weight' % i: distil_model.peer_weights[i].item(),
                                              'teacher%d_weight_pre' % i: predistil_model.peer_weights[i].item(),
                                              'teacher%d_weight_adv' % i: advdistil_model.peer_weights[i].item(),
                                              'teacher%d_weight_softmax' % i: distil_model.peer_weights_softmax[i].item(),
                                              'teacher%d_weight_softmax_pre' % i: predistil_model.peer_weights_softmax[i].item(),
                                              'teacher%d_weight_softmax_adv' % i: advdistil_model.peer_weights_softmax[i].item()
                                              } for i, teacher in enumerate(teachers)]

                    wandb_log = wandb_log + [{'teacher%d_loss_clm' % i: teacher['loss_clm'].item(),
                                              'teacher%d_acc' % i: teacher['acc'],
                                              'teacher%d_target_acc' % i: teacher['target_acc']
                                              } for i, teacher in enumerate(teachers)]

                    wandb_log = wandb_log + [{'teacher%d_dir_loss_clm' % i: teacher['dir_loss_clm'].item(),
                                              'teacher%d_dir_acc' % i: teacher['dir_acc'],
                                              'teacher%d_dir_target_acc' % i: teacher['dir_target_acc']
                                              } for i, teacher in enumerate(teachers[:-1])]

                    wandb.log({k: v for d in wandb_log for k, v in d.items()})

            torch.cuda.empty_cache()

            all_losses = (distil_total_loss +
                          predistil_total_loss +
                          advdistil_total_loss +
                          undistil_loss +
                          teacher_losses)

            all_losses.backward()  # accumulate gradients wrt training loss

            if epoch % config.nucleus.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(distil_model.parameters(), 0.5)
                distil_optimizer.step()  # update model parameters to reduce loss
                distil_optimizer.zero_grad()  # remove previously accumulated gradients
                if distil_scheduler:
                    distil_scheduler.step()  # update learning rate multiplier

                # Unused: opting for using main optimizer for peer-weights also
                # distil_weight_optimizer.step()  # update model parameters to reduce loss
                # distil_weight_optimizer.zero_grad()  # remove previously accumulated gradients
                # distil_weight_scheduler.step()

                torch.nn.utils.clip_grad_norm_(predistil_model.parameters(), 0.5)
                predistil_optimizer.step()  # update model parameters to reduce loss
                predistil_optimizer.zero_grad()  # remove previously accumulated gradients
                if predistil_scheduler:
                    predistil_scheduler.step()  # update learning rate multiplier

                torch.nn.utils.clip_grad_norm_(advdistil_model.parameters(), 0.5)
                advdistil_optimizer.step()  # update model parameters to reduce loss
                advdistil_optimizer.zero_grad()  # remove previously accumulated gradients
                if advdistil_scheduler:
                    advdistil_scheduler.step()  # update learning rate multiplier

                torch.nn.utils.clip_grad_norm_(undistil_model.parameters(), 0.5)
                undistil_optimizer.step()  # update model parameters to reduce loss
                undistil_optimizer.zero_grad()  # remove previously accumulated gradients
                if undistil_scheduler:
                    undistil_scheduler.step()  # update learning rate multiplier

                adaptor_optimizer.step()
                adaptor_optimizer.zero_grad()
                adaptor_scheduler.step()

            torch.cuda.empty_cache()


if __name__ == '__main__':
    use_config = main_config()
    main(use_config)
