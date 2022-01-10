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

import wandb
import torch
import argparse
import bittensor
import transformers

from datasets import load_dataset
from types import SimpleNamespace
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import GPT2TokenizerFast, GPT2LMHeadModel

# from bittensor._neuron.text.template_miner.nucleus_impl import Nucleus
from nucleus_300_2 import Nucleus
from nucleus_300_3 import Nucleus as Nucleus3


def modify_args(parser: argparse.ArgumentParser):
    r""" Modify custom params in the parser for this test.
    """
    parser.add_argument('--wandb.name', type=str, help='''Optionally pass wandb run name for use_wandb''',
                        default='BIT-300-sgmoe-3')
    parser.add_argument('--wandb.project', type=str, help='''Optionally pass wandb project name for use_wandb''',
                        default='neuron-tests-sgmoe')
    parser.add_argument('--wandb.tags', type=str, help='''Optionally pass wandb tags for use_wandb''',
                        default='hf losses, no-pos-enc, neuron, test, template_miner_distil, gpt2, '
                                'remotes weighted-join, mixture-of-experts, sparse routing, synergy')
    parser.add_argument('--wandb.run_group', type=str, help='''Optionally pass wandb group name for use_wandb''',
                        default='template_miner_sgmoe-3')

    parser.add_argument('--dataset.batch_size', type=int, help='Batch size.', default=16)
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
                        help='the number of nn.TransformerEncoderLayer in nn.TransformerEncoder', default=12)
    parser.add_argument('--nucleus.nlayers_local_hidden', type=int,
                        help='the number of nn.TransformerEncoderLayer in nn.TransformerEncoder', default=2)
    parser.add_argument('--nucleus.nlayers_remote_hidden', type=int,
                        help='the number of nn.TransformerEncoderLayer in nn.TransformerEncoder', default=2)
    parser.add_argument('--nucleus.dropout', type=float, help='the dropout value', default=0.1)

    # From: https://github.com/huggingface/transformers/blob/master/examples/research_projects/distillation/train.py
    parser.add_argument(
        "--nucleus.gradient_accumulation_steps",
        type=int,
        default=4,
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

    parser.add_argument('--neuron.learning_rate', type=float, help='Training initial learning rate.', default=1e-4)
    parser.add_argument('--neuron.weight_decay', type=float, help='nucleus parameter weight decay.', default=0.25)
    parser.add_argument('--neuron.momentum', type=float, help='optimizer momentum.', default=0.8)
    parser.add_argument('--neuron.clip_gradients', type=float,
                        help='Implement gradient clipping to avoid exploding loss on smaller architectures.',
                        default=1.0)
    parser.add_argument('--neuron.batch_size_train', type=int, help='Training batch size.', default=32)
    parser.add_argument('--neuron.device', type=str, help='Torch device for training.', default='cuda:1')
    parser.add_argument('--neuron.second_device', type=str, help='Torch second device training.',
                        default='cuda:1')
    parser.add_argument('--neuron.use_wandb', action='store_true',
                        help='''neuron activates its weights and biases powers''', default=False)
    parser.add_argument('--neuron.n_epochs', type=int, help='Number of training epochs.', default=900000)
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


class Random(nn.Module):
    def __init__(self):
        super(Random, self).__init__()

        self.embedding = nn.Embedding(bittensor.__vocab_size__, bittensor.__network_dim__)
        self.decoder = nn.Linear(bittensor.__network_dim__, bittensor.__vocab_size__)

    def forward(self, input_ids: torch.LongTensor, output_hidden_states: bool = False) -> SimpleNamespace:
        output = SimpleNamespace()
        # embedding: retrieve learned representation vectors for input vocabulary tokens.
        # inputs.shape = [batch_size, sequence_len]
        # embedding.shape = [batch_size, sequence_len, bittensor.__network_dim__]
        output.local_context = self.embedding(input_ids)
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
                {'name': 'distilgpt2',  # 6-layer, 768-hidden, 12-heads, 82M parameters
                 'dim': 768,
                 'device': config.neuron.device},
                {'name': 'gpt2',  # 12-layer, 768-hidden, 12-heads, 117M parameters.
                 'dim': 768,
                 'device': config.neuron.device},
                {'name': 'gpt2-medium',  # 24-layer, 1024-hidden, 16-heads, 345M parameters.
                 'dim': 1024,
                 'device': config.neuron.device},
                {'name': 'gpt2-large',  # 36-layer, 1280-hidden, 20-heads, 774M parameters.
                 'dim': 1280,
                 'device': config.neuron.second_device},
                {'name': 'gpt2-xl',  # 48-layer, 1600-hidden, 25-heads, 1558M parameters.
                 'dim': 1600,
                 'device': config.neuron.second_device},
                {'name': 'random',  # random encoder
                 'dim': 1024,
                 'device': config.neuron.device,
                 'model': Random().half().to(config.neuron.device)}
                ]

    for teacher in teachers[:-1]:
        # Load pretrained teacher models with language-modeling heads
        teacher['model'] = GPT2LMHeadModel.from_pretrained(teacher['name']).half().to(teacher['device'])

    for teacher in teachers:
        # Adapt the teacher hidden dimension to the bittensor network dimension of bittensor by using
        #  a fully-connected layer to convert teacher hidden features to a size of bittensor.__network_dim__
        teacher['adaptor'] = nn.Linear(teacher['dim'], bittensor.__network_dim__, bias=False).to(distil_device)
        teacher['adaptor2'] = nn.Linear(teacher['dim'], bittensor.__network_dim__, bias=False).to(distil_device)
        teacher['adaptor3'] = nn.Linear(teacher['dim'], bittensor.__network_dim__, bias=False).to(distil_device)
        teacher['adaptor4'] = nn.Linear(teacher['dim'], bittensor.__network_dim__, bias=False).to(distil_device)

    # Learn the dimension adaptors for the input teachers getting mixed
    adaptor_params = sum((list(teacher['adaptor'].parameters()) for teacher in teachers), [])
    adaptor_optimizer = torch.optim.AdamW(adaptor_params, lr=config.neuron.learning_rate)

    adaptor2_params = sum((list(teacher['adaptor2'].parameters()) for teacher in teachers), [])
    adaptor2_optimizer = torch.optim.AdamW(adaptor2_params, lr=config.neuron.learning_rate)

    adaptor3_params = sum((list(teacher['adaptor3'].parameters()) for teacher in teachers), [])
    adaptor3_optimizer = torch.optim.AdamW(adaptor3_params, lr=config.neuron.learning_rate)

    adaptor4_params = sum((list(teacher['adaptor4'].parameters()) for teacher in teachers), [])
    adaptor4_optimizer = torch.optim.AdamW(adaptor4_params, lr=config.neuron.learning_rate)

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
    distilb_device = config.neuron.device
    distilb_config = config.copy()
    distilb_config.neuron.device = distilb_device
    distilb_config.nucleus.alpha_clm = 1.
    distilb_config.nucleus.alpha_clm_dis = 0.0
    distilb_config.nucleus.alpha_clm_rmt = 1.0
    distilb_config.nucleus.alpha_mse = 0.0
    distilb_config.nucleus.alpha_mse_hid = 1.0
    distilb_config.nucleus.alpha_ce = 0.0
    distilb_config.nucleus.alpha_cos = 0.0
    distilb_model = Nucleus(distilb_config)
    distilb_model.peer_weights = nn.Parameter(torch.ones([len(teachers)], requires_grad=True, device=distilb_device))
    # Load same initialization as distil_model
    distilb_model.load_state_dict(distil_state, strict=True)
    distilb_model = distilb_model.to(distilb_device)

    print(distilb_model)
    print('distilb', distilb_model.alpha_ce, distilb_model.alpha_clm, distilb_model.alpha_clm_dis,
          distilb_model.alpha_clm_rmt, distilb_model.alpha_mse, distilb_model.alpha_mse_hid,
          distilb_model.alpha_cos)

    # # Initialize another nucleus that distils from an annealed sgmoe
    # predistil_device = config.neuron.device
    # predistil_config = config.copy()
    # predistil_config.neuron.device = predistil_device
    # predistil_config.nucleus.alpha_clm = 1.
    # predistil_config.nucleus.alpha_clm_dis = 0.0
    # predistil_config.nucleus.alpha_clm_rmt = 1.0
    # predistil_config.nucleus.alpha_mse = 0.0
    # predistil_config.nucleus.alpha_mse_hid = 1.0
    # predistil_config.nucleus.alpha_ce = 0.0
    # predistil_config.nucleus.alpha_cos = 0.0
    # predistil_model = Nucleus3(predistil_config)
    # predistil_model.peer_weights = nn.Parameter(
    #     torch.ones([len(teachers)], requires_grad=True, device=predistil_device))
    # # Load same initialization as distil_model
    # predistil_model.load_state_dict(distil_state, strict=True)
    # predistil_model = predistil_model.to(predistil_device)
    #
    # print(predistil_model)
    # print('predistil', predistil_model.alpha_ce, predistil_model.alpha_clm, predistil_model.alpha_clm_dis,
    #       predistil_model.alpha_clm_rmt, predistil_model.alpha_mse, predistil_model.alpha_mse_hid,
    #       predistil_model.alpha_cos)

    # Initialize another nucleus that distils from an annealed sgmoe
    predistil1_device = config.neuron.device
    predistil1_config = config.copy()
    predistil1_config.neuron.device = predistil1_device
    predistil1_config.nucleus.alpha_clm = 1.
    predistil1_config.nucleus.alpha_clm_dis = 0.0
    predistil1_config.nucleus.alpha_clm_rmt = 1.0
    predistil1_config.nucleus.alpha_mse = 0.0
    predistil1_config.nucleus.alpha_mse_hid = 1.0
    predistil1_config.nucleus.alpha_ce = 0.0
    predistil1_config.nucleus.alpha_cos = 0.0
    predistil1_model = Nucleus3(predistil1_config)
    predistil1_model.peer_weights = nn.Parameter(
        torch.ones([len(teachers)], requires_grad=True, device=predistil1_device))
    # Load same initialization as distil_model
    predistil1_model.load_state_dict(distil_state, strict=True)
    predistil1_model = predistil1_model.to(predistil1_device)

    print(predistil1_model)
    print('predistil1', predistil1_model.alpha_ce, predistil1_model.alpha_clm, predistil1_model.alpha_clm_dis,
          predistil1_model.alpha_clm_rmt, predistil1_model.alpha_mse, predistil1_model.alpha_mse_hid,
          predistil1_model.alpha_cos)

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
    distilb_optimizer = torch.optim.AdamW(distilb_model.parameters(), lr=config.neuron.learning_rate)
    # predistil_optimizer = torch.optim.AdamW(predistil_model.parameters(), lr=config.neuron.learning_rate)
    predistil1_optimizer = torch.optim.AdamW(predistil1_model.parameters(), lr=config.neuron.learning_rate)
    undistil_optimizer = torch.optim.AdamW(undistil_model.parameters(), lr=config.neuron.learning_rate)

    # Define learning rate scheduler (multiplier) for optimizer
    distil_scheduler = None
    distilb_scheduler = None
    # predistil_scheduler = None
    predistil1_scheduler = None
    undistil_scheduler = None
    adaptor_scheduler = None

    if config.neuron.lr_scheduler == 'get_cosine_schedule_with_warmup':
        adaptor_scheduler = transformers.get_cosine_schedule_with_warmup(optimizer=adaptor_optimizer,
                                                                         num_warmup_steps=config.neuron.num_warmup_steps,
                                                                         num_training_steps=config.neuron.n_epochs)
        distil_scheduler = transformers.get_cosine_schedule_with_warmup(optimizer=distil_optimizer,
                                                                        num_warmup_steps=config.neuron.num_warmup_steps,
                                                                        num_training_steps=config.neuron.n_epochs)
        distilb_scheduler = transformers.get_cosine_schedule_with_warmup(optimizer=distilb_optimizer,
                                                                          num_warmup_steps=config.neuron.num_warmup_steps,
                                                                          num_training_steps=config.neuron.n_epochs)
        # predistil_scheduler = transformers.get_cosine_schedule_with_warmup(optimizer=predistil_optimizer,
        #                                                                   num_warmup_steps=config.neuron.num_warmup_steps,
        #                                                                   num_training_steps=config.neuron.n_epochs)
        predistil1_scheduler = transformers.get_cosine_schedule_with_warmup(optimizer=predistil1_optimizer,
                                                                          num_warmup_steps=config.neuron.num_warmup_steps,
                                                                          num_training_steps=config.neuron.n_epochs)
        undistil_scheduler = transformers.get_cosine_schedule_with_warmup(optimizer=undistil_optimizer,
                                                                          num_warmup_steps=config.neuron.num_warmup_steps,
                                                                          num_training_steps=config.neuron.n_epochs)

    elif config.neuron.lr_scheduler == 'get_cosine_with_hard_restarts_schedule_with_warmup':
        adaptor_scheduler = transformers.get_cosine_with_hard_restarts_schedule_with_warmup(optimizer=adaptor_optimizer,
                                                                                            num_warmup_steps=config.neuron.num_warmup_steps,
                                                                                            num_training_steps=config.neuron.n_epochs,
                                                                                            num_cycles=config.neuron.num_cycles)
        distil_scheduler = transformers.get_cosine_with_hard_restarts_schedule_with_warmup(optimizer=distil_optimizer,
                                                                                           num_warmup_steps=config.neuron.num_warmup_steps,
                                                                                           num_training_steps=config.neuron.n_epochs,
                                                                                           num_cycles=config.neuron.num_cycles)
        distilb_scheduler = transformers.get_cosine_with_hard_restarts_schedule_with_warmup(
            optimizer=distilb_optimizer,
            num_warmup_steps=config.neuron.num_warmup_steps,
            num_training_steps=config.neuron.n_epochs,
            num_cycles=config.neuron.num_cycles)
        # predistil_scheduler = transformers.get_cosine_with_hard_restarts_schedule_with_warmup(
        #     optimizer=predistil_optimizer,
        #     num_warmup_steps=config.neuron.num_warmup_steps,
        #     num_training_steps=config.neuron.n_epochs,
        #     num_cycles=config.neuron.num_cycles)
        predistil1_scheduler = transformers.get_cosine_with_hard_restarts_schedule_with_warmup(
            optimizer=predistil1_optimizer,
            num_warmup_steps=config.neuron.num_warmup_steps,
            num_training_steps=config.neuron.n_epochs,
            num_cycles=config.neuron.num_cycles)
        undistil_scheduler = transformers.get_cosine_with_hard_restarts_schedule_with_warmup(
            optimizer=undistil_optimizer,
            num_warmup_steps=config.neuron.num_warmup_steps,
            num_training_steps=config.neuron.n_epochs,
            num_cycles=config.neuron.num_cycles)

    if config.neuron.use_wandb:
        bittensor.wandb(config)  # Initialize wandb logging
        wandb.watch(distil_model)  # Track model parameters and gradients
        wandb.watch(distilb_model)  # Track model parameters and gradients
        # wandb.watch(predistil_model)  # Track model parameters and gradients
        wandb.watch(predistil1_model)  # Track model parameters and gradients
        wandb.watch(undistil_model)  # Track model parameters and gradients
        for teacher in teachers:
            wandb.watch(teacher['adaptor'])
            wandb.watch(teacher['adaptor2'])
            wandb.watch(teacher['adaptor3'])
            wandb.watch(teacher['adaptor4'])
        wandb_table_data = []

    # how many experts/teachers per batch sequence
    # annealed from 1 at the start to all teachers over time
    # experts_per_input = None
    # experts_schedule = {0: 1, 100000: 2, 200000: 3}

    for dataset_epoch in range(3):
        # Define pytorch dataloader with shuffled batches of batch_size token sequences of block_size length.
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch, batch in enumerate(dataloader):
            # if epoch in experts_schedule:
            #     experts_per_input = experts_schedule[epoch]

            with torch.no_grad():
                input_ids = batch['input_ids'].to(distil_device)
                target = input_ids[:, -1]  # held out target of last token
                input_ids = input_ids[:, :-1]  # entire sequence except last token

                teacher_inputs = {}

                for teacher in teachers:
                    if teacher['device'] not in teacher_inputs:
                        teacher_inputs[teacher['device']] = input_ids.clone().to(teacher['device'])

                    teacher_input_ids = teacher_inputs[teacher['device']]
                    teacher_output = teacher['model'](input_ids=teacher_input_ids, output_hidden_states=True)
                    teacher['hidden_states'] = teacher_output.hidden_states[-1].float()

                    shift_logits = teacher_output.logits[..., :-1, :].float().contiguous()
                    shift_labels = teacher_input_ids[..., 1:].contiguous()
                    teacher['loss_clm'] = nn.CrossEntropyLoss()(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                    predictions = shift_logits.detach().max(2).indices
                    teacher['acc'] = (predictions == shift_labels).sum().item() / predictions.nelement()

                    teacher['prediction'] = teacher_output.logits[:, -1, :].argmax(-1)  # predict unseen last token
                    teacher['predictions'] = tokenizer.decode(teacher_output.logits[0].argmax(-1).detach())

                    teacher_target = target.clone().to(teacher['device'])
                    teacher['target_acc'] = (teacher['prediction'] == teacher_target).sum().item() / len(teacher_target)

                adaptor_lr = adaptor_optimizer.param_groups[0]['lr']

            # Weighted joining of teachers with weights that also get learned
            joining_weights = F.softmax(distil_model.peer_weights, dim=0)
            distil_teacher_inputs = None
            for i, teacher in enumerate(teachers):
                if distil_teacher_inputs is None:
                    distil_teacher_inputs = joining_weights[i] * teacher['adaptor'](
                        teacher['hidden_states'].detach().to(distil_device))
                else:
                    distil_teacher_inputs += joining_weights[i] * teacher['adaptor'](
                        teacher['hidden_states'].detach().to(distil_device))

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

                distilb_input_ids = input_ids.detach().to(distilb_device)

            # only use GPT2-XL as teacher - to compare vs mix for synergy
            teacher = teachers[4]
            distilb_teacher_inputs = teacher['adaptor2'](
                    teacher['hidden_states'].detach().to(distilb_device))

            distilb_output = distilb_model.remote_forward(distilb_input_ids, training=True,
                                                          teacher_inputs=distilb_teacher_inputs)  # forward pass in transformer model
            distilb_total_loss = (distilb_model.alpha_clm * distilb_output.loss_clm +
                                  distilb_model.alpha_clm_dis * distilb_output.loss_clm_dis +
                                  distilb_model.alpha_clm_rmt * distilb_output.loss_clm_rmt +
                                  distilb_model.alpha_mse * distilb_output.loss_mse +
                                  distilb_model.alpha_mse_hid * distilb_output.loss_mse_hid +
                                  distilb_model.alpha_ce * distilb_output.loss_ce +
                                  distilb_model.alpha_cos * distilb_output.loss_cos)

            with torch.no_grad():
                distilb_loss_clm = distilb_output.loss_clm
                distilb_loss_clm_dis = distilb_output.loss_clm_dis
                distilb_loss_clm_rmt = distilb_output.loss_clm_rmt
                distilb_loss_mse = distilb_output.loss_mse
                distilb_loss_mse_hid = distilb_output.loss_mse_hid
                distilb_loss_ce = distilb_output.loss_ce
                distilb_loss_cos = distilb_output.loss_cos
                distilb_acc = distilb_output.local_accuracy  # training accuracy on next token prediction in train sequence with masking
                distilb_remote_acc = distilb_output.remote_accuracy  # training accuracy on next token prediction in train sequence with masking
                distilb_lr = distilb_optimizer.param_groups[0]['lr']  # record actual learning rate
                # distilb_weight_lr = distilb_weight_optimizer.param_groups[0]['lr']  # record actual learning rate

                distilb_target = target.to(distilb_device)
                distilb_prediction = distilb_output.local_target[:, -1, :].argmax(-1)  # predict unseen last token
                distilb_target_acc = (distilb_prediction == distilb_target).sum().item() / len(
                    distilb_target)  # validation accuracy on predicting unseen token

                distilb_remote_prediction = distilb_output.remote_target[:, -1, :].argmax(
                    -1)  # predict unseen last token
                distilb_remote_target_acc = (distilb_remote_prediction == distilb_target).sum().item() / len(
                    distilb_target)  # validation accuracy on predicting unseen token

            #     predistil_input_ids = input_ids.detach().to(predistil_device)
            #
            # # Weighted joining of teachers with weights that also get learned
            # joining_weights = F.softmax(predistil_model.peer_weights, dim=0)
            # predistil_teacher_inputs = []
            # for i, teacher in enumerate(teachers):
            #     predistil_teacher_inputs += [teacher['adaptor3'](
            #         teacher['hidden_states'].detach().to(predistil_device))]
            #
            # # [expert_len * batch_size, sequence_len, bittensor.__network_dim__]
            # predistil_teacher_inputs = torch.cat(predistil_teacher_inputs)
            #
            # predistil_output = predistil_model.remote_forward(predistil_input_ids, training=True,
            #                                                   teacher_inputs=predistil_teacher_inputs,
            #                                                   join_weights=joining_weights)  # forward pass in local transformer model
            # predistil_total_loss = (predistil_model.alpha_clm * predistil_output.loss_clm +
            #                         predistil_model.alpha_clm_dis * predistil_output.loss_clm_dis +
            #                         predistil_model.alpha_clm_rmt * predistil_output.loss_clm_rmt +
            #                         predistil_model.alpha_mse * predistil_output.loss_mse +
            #                         predistil_model.alpha_mse_hid * predistil_output.loss_mse_hid +
            #                         predistil_model.alpha_ce * predistil_output.loss_ce +
            #                         predistil_model.alpha_cos * predistil_output.loss_cos)
            #
            # with torch.no_grad():
            #     predistil_loss_clm = predistil_output.loss_clm
            #     predistil_loss_clm_dis = predistil_output.loss_clm_dis
            #     predistil_loss_clm_rmt = predistil_output.loss_clm_rmt
            #     predistil_loss_mse = predistil_output.loss_mse
            #     predistil_loss_mse_hid = predistil_output.loss_mse_hid
            #     predistil_loss_ce = predistil_output.loss_ce
            #     predistil_loss_cos = predistil_output.loss_cos
            #     predistil_acc = predistil_output.local_accuracy  # training accuracy on next token prediction in train sequence with masking
            #     predistil_remote_acc = predistil_output.remote_accuracy  # training accuracy on next token prediction in train sequence with masking
            #     predistil_lr = predistil_optimizer.param_groups[0]['lr']  # record actual learning rate
            #     # predistil_weight_lr = predistil_weight_optimizer.param_groups[0]['lr']  # record actual learning rate
            #
            #     predistil_target = target.to(predistil_device)
            #     predistil_prediction = predistil_output.local_target[:, -1, :].argmax(-1)  # predict unseen last token
            #     predistil_target_acc = (predistil_prediction == predistil_target).sum().item() / len(
            #         predistil_target)  # validation accuracy on predicting unseen token
            #
            #     # batch_weights.shape = [expert_len * batch_size]
            #     batch_weights = joining_weights.repeat_interleave(batch_size)
            #
            #     # predistil_remote_prediction.shape = [expert_len * batch_weights]
            #     predistil_remote_prediction = predistil_output.remote_target[:, -1, :].argmax(
            #         -1)  # predict unseen last token
            #     predistil_target_expert = predistil_target.repeat(len(teachers))
            #     predistil_remote_target_acc = (batch_weights * (
            #                 predistil_remote_prediction == predistil_target_expert).float()).sum().item() / len(
            #         predistil_target)  # validation accuracy on predicting unseen token

                predistil1_input_ids = input_ids.detach().to(predistil1_device)

                ratio = torch.tensor(0.01)  # ensure min teacher contributes at least 10% to mix
                min_weight = predistil1_model.peer_weights.min()
                max_weight = predistil1_model.peer_weights.max()
                if torch.exp(min_weight) / torch.exp(max_weight) < ratio:
                    reweight_factor = torch.log(ratio) / (min_weight - max_weight)
                    predistil1_model.peer_weights *= reweight_factor

            # Weighted joining of teachers with weights that also get learned
            joining_weights = F.softmax(predistil1_model.peer_weights, dim=0)
            predistil1_teacher_inputs = []
            for i, teacher in enumerate(teachers):
                predistil1_teacher_inputs += [teacher['adaptor4'](
                    teacher['hidden_states'].detach().to(predistil1_device))]

            # [expert_len * batch_size, sequence_len, bittensor.__network_dim__]
            predistil1_teacher_inputs = torch.cat(predistil1_teacher_inputs)

            predistil1_output = predistil1_model.remote_forward(predistil1_input_ids, training=True,
                                                              teacher_inputs=predistil1_teacher_inputs,
                                                              join_weights=joining_weights)  # forward pass in local transformer model
            predistil1_total_loss = (predistil1_model.alpha_clm * predistil1_output.loss_clm +
                                    predistil1_model.alpha_clm_dis * predistil1_output.loss_clm_dis +
                                    predistil1_model.alpha_clm_rmt * predistil1_output.loss_clm_rmt +
                                    predistil1_model.alpha_mse * predistil1_output.loss_mse +
                                    predistil1_model.alpha_mse_hid * predistil1_output.loss_mse_hid +
                                    predistil1_model.alpha_ce * predistil1_output.loss_ce +
                                    predistil1_model.alpha_cos * predistil1_output.loss_cos)

            with torch.no_grad():
                predistil1_loss_clm = predistil1_output.loss_clm
                predistil1_loss_clm_dis = predistil1_output.loss_clm_dis
                predistil1_loss_clm_rmt = predistil1_output.loss_clm_rmt
                predistil1_loss_mse = predistil1_output.loss_mse
                predistil1_loss_mse_hid = predistil1_output.loss_mse_hid
                predistil1_loss_ce = predistil1_output.loss_ce
                predistil1_loss_cos = predistil1_output.loss_cos
                predistil1_acc = predistil1_output.local_accuracy  # training accuracy on next token prediction in train sequence with masking
                predistil1_remote_acc = predistil1_output.remote_accuracy  # training accuracy on next token prediction in train sequence with masking
                predistil1_lr = predistil1_optimizer.param_groups[0]['lr']  # record actual learning rate
                # predistil1_weight_lr = predistil1_weight_optimizer.param_groups[0]['lr']  # record actual learning rate

                predistil1_target = target.to(predistil1_device)
                predistil1_prediction = predistil1_output.local_target[:, -1, :].argmax(-1)  # predict unseen last token
                predistil1_target_acc = (predistil1_prediction == predistil1_target).sum().item() / len(
                    predistil1_target)  # validation accuracy on predicting unseen token

                # batch_weights.shape = [expert_len * batch_size]
                batch_weights = joining_weights.repeat_interleave(batch_size)

                # predistil1_remote_prediction.shape = [expert_len * batch_weights]
                predistil1_remote_prediction = predistil1_output.remote_target[:, -1, :].argmax(
                    -1)  # predict unseen last token
                predistil1_target_expert = predistil1_target.repeat(len(teachers))
                predistil1_remote_target_acc = (batch_weights * (
                            predistil1_remote_prediction == predistil1_target_expert).float()).sum().item() / len(
                    predistil1_target)  # validation accuracy on predicting unseen token

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
                                               predistil1_total_loss.item(),
                                               distil_acc, predistil1_acc,
                                               distil_target_acc, distil_remote_target_acc,
                                               teachers[-1]['target_acc'], predistil1_target_acc,
                                               distil_lr), end=' ')

                if epoch % 1000 == 0:
                    input_decoded = tokenizer.decode(input_ids[0])
                    distil_predictions = distil_output.local_target[0].detach().argmax(-1)
                    predistil1_predictions = predistil1_output.local_target[0].detach().argmax(-1)

                    print('\n.\n', input_decoded, '\n...\n')
                    print(list(zip([tokenizer.decode(_) for _ in input_ids[0]],
                                   [tokenizer.decode(_) for _ in distil_predictions])), '\n.\n')

                    distil_predictions = tokenizer.decode(distil_predictions)
                    predistil1_predictions = tokenizer.decode(predistil1_predictions)
                    if config.neuron.use_wandb:
                        wandb_table_data += [[epoch,
                                              distil_target_acc,
                                              distil_predictions, predistil1_predictions, input_decoded] +
                                             [teacher['predictions'] for teacher in teachers]]

                if config.neuron.use_wandb:
                    if epoch % 5000 == 0:
                        wandb_table = wandb.Table(columns=['epoch',
                                                           'distil_target_acc',
                                                           'distil_predictions', 'predistil1_predictions', 'input'] +
                                                          ['%s' % teacher['name'] for teacher in teachers])
                        for row in wandb_table_data:
                            wandb_table.add_data(*row)

                        torch.save(distil_model.state_dict(), "{}/distil_model_sgmoe2.torch".format(config.wandb.directory))

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

                                 'distilb_loss_clm': distilb_loss_clm.item(),
                                 'distilb_loss_clm_dis': distilb_loss_clm_dis.item(),
                                 'distilb_loss_clm_rmt': distilb_loss_clm_rmt.item(),
                                 'distilb_loss_mse': distilb_loss_mse.item(),
                                 'distilb_loss_mse_hid': distilb_loss_mse_hid.item(),
                                 'distilb_loss_ce': distilb_loss_ce.item(),
                                 'distilb_loss_cos': distilb_loss_cos.item(),
                                 'distilb_total_loss': distilb_total_loss.item(),
                                 'distilb_acc': distilb_acc,
                                 'distilb_remote_acc': distilb_remote_acc,
                                 'distilb_target_acc': distilb_target_acc,
                                 'distilb_remote_target_acc': distilb_remote_target_acc,
                                 'distilb_lr': distilb_lr,
                                 'distilb_expert_calls': batch_size * epoch,

                                 # 'predistil_loss_clm': predistil_loss_clm.item(),
                                 # 'predistil_loss_clm_dis': predistil_loss_clm_dis.item(),
                                 # 'predistil_loss_clm_rmt': predistil_loss_clm_rmt.item(),
                                 # 'predistil_loss_mse': predistil_loss_mse.item(),
                                 # 'predistil_loss_mse_hid': predistil_loss_mse_hid.item(),
                                 # 'predistil_loss_ce': predistil_loss_ce.item(),
                                 # 'predistil_loss_cos': predistil_loss_cos.item(),
                                 # 'predistil_total_loss': predistil_total_loss.item(),
                                 # 'predistil_acc': predistil_acc,
                                 # 'predistil_remote_acc': predistil_remote_acc,
                                 # 'predistil_target_acc': predistil_target_acc,
                                 # 'predistil_remote_target_acc': predistil_remote_target_acc,
                                 # 'predistil_lr': predistil_lr,
                                 # 'predistil_expert_calls': len(teachers) * batch_size * epoch,

                                 'predistil1_loss_clm': predistil1_loss_clm.item(),
                                 'predistil1_loss_clm_dis': predistil1_loss_clm_dis.item(),
                                 'predistil1_loss_clm_rmt': predistil1_loss_clm_rmt.item(),
                                 'predistil1_loss_mse': predistil1_loss_mse.item(),
                                 'predistil1_loss_mse_hid': predistil1_loss_mse_hid.item(),
                                 'predistil1_loss_ce': predistil1_loss_ce.item(),
                                 'predistil1_loss_cos': predistil1_loss_cos.item(),
                                 'predistil1_total_loss': predistil1_total_loss.item(),
                                 'predistil1_acc': predistil1_acc,
                                 'predistil1_remote_acc': predistil1_remote_acc,
                                 'predistil1_target_acc': predistil1_target_acc,
                                 'predistil1_remote_target_acc': predistil1_remote_target_acc,
                                 'predistil1_lr': predistil1_lr,
                                 'predistil1_expert_calls': len(teachers) * batch_size * epoch,

                                 'undistil_loss': undistil_loss.item(),
                                 'undistil_acc': undistil_acc,
                                 'undistil_target_acc': undistil_target_acc,
                                 'undistil_lr': undistil_lr,

                                 'adaptor_lr': adaptor_lr,
                                 # 'experts_per_input': experts_per_input,
                                 'batch': epoch}

                    wandb_log = [wandb_log] + [{'teacher%d_weight' % i: distil_model.peer_weights[i].item(),
                                                # 'teacher%d_weight_pre' % i: predistil_model.peer_weights[i].item(),
                                                'teacher%d_weight_pre1' % i: predistil1_model.peer_weights[i].item(),
                                                'teacher%d_loss_clm' % i: teacher['loss_clm'].item(),
                                                'teacher%d_acc' % i: teacher['acc'],
                                                'teacher%d_target_acc' % i: teacher['target_acc'],
                                                } for i, teacher in enumerate(teachers)]

                    wandb.log({k: v for d in wandb_log for k, v in d.items()})

            torch.cuda.empty_cache()

            distil_total_loss.backward()  # accumulate gradients wrt training loss
            distilb_total_loss.backward()  # accumulate gradients wrt training loss
            # predistil_total_loss.backward()  # accumulate gradients wrt training loss
            predistil1_total_loss.backward()  # accumulate gradients wrt training loss
            undistil_loss.backward()  # accumulate gradients wrt training loss

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

                torch.nn.utils.clip_grad_norm_(distilb_model.parameters(), 0.5)
                distilb_optimizer.step()  # update model parameters to reduce loss
                distilb_optimizer.zero_grad()  # remove previously accumulated gradients
                if distilb_scheduler:
                    distilb_scheduler.step()  # update learning rate multiplier

                # torch.nn.utils.clip_grad_norm_(predistil_model.parameters(), 0.5)
                # predistil_optimizer.step()  # update model parameters to reduce loss
                # predistil_optimizer.zero_grad()  # remove previously accumulated gradients
                # if predistil_scheduler:
                #     predistil_scheduler.step()  # update learning rate multiplier

                torch.nn.utils.clip_grad_norm_(predistil1_model.parameters(), 0.5)
                predistil1_optimizer.step()  # update model parameters to reduce loss
                predistil1_optimizer.zero_grad()  # remove previously accumulated gradients
                if predistil1_scheduler:
                    predistil1_scheduler.step()  # update learning rate multiplier

                torch.nn.utils.clip_grad_norm_(undistil_model.parameters(), 0.5)
                undistil_optimizer.step()  # update model parameters to reduce loss
                undistil_optimizer.zero_grad()  # remove previously accumulated gradients
                if undistil_scheduler:
                    undistil_scheduler.step()  # update learning rate multiplier

                adaptor_optimizer.step()
                adaptor_optimizer.zero_grad()
                adaptor2_optimizer.step()
                adaptor2_optimizer.zero_grad()
                adaptor3_optimizer.step()
                adaptor3_optimizer.zero_grad()
                adaptor4_optimizer.step()
                adaptor4_optimizer.zero_grad()
                adaptor_scheduler.step()

            torch.cuda.empty_cache()


if __name__ == '__main__':
    use_config = main_config()
    main(use_config)
