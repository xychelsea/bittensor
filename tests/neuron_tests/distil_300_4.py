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
from transformers import AutoTokenizer, AutoModelForCausalLM

# from bittensor._neuron.text.template_miner.nucleus_impl import Nucleus
from nucleus_300_2 import Nucleus
from nucleus_300_4 import Nucleus as Nucleus4


def modify_args(parser: argparse.ArgumentParser):
    r""" Modify custom params in the parser for this test.
    """
    parser.add_argument('--wandb.name', type=str, help='''Optionally pass wandb run name for use_wandb''',
                        default='BIT-300-sgmoe-4')
    parser.add_argument('--wandb.project', type=str, help='''Optionally pass wandb project name for use_wandb''',
                        default='neuron-tests-sgmoe')
    parser.add_argument('--wandb.tags', type=str, help='''Optionally pass wandb tags for use_wandb''',
                        default='hf losses, no-pos-enc, neuron, test, template_miner_distil, gpt2, '
                                'remotes weighted-join, mixture-of-experts, sparse routing, gpt2-german, tokenizer misalign')
    parser.add_argument('--wandb.run_group', type=str, help='''Optionally pass wandb group name for use_wandb''',
                        default='template_miner_sgmoe-4')

    parser.add_argument('--dataset.batch_size', type=int, help='Batch size.', default=16)
    parser.add_argument('--dataset.block_size', type=int, help='Number of text items to pull for each example..',
                        default=80)
    parser.add_argument('--dataset.num_workers', type=int, help='Number of workers for data loader.', default=20)
    parser.add_argument('--dataset.name', type=str, help='Which dataset to use.', default='wikipedia')
    parser.add_argument('--dataset.subset', type=str, help='Which dataset to use.', default='20200501.de')
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

    parser.add_argument('--neuron.learning_rate', type=float, help='Training initial learning rate.', default=7e-5)
    parser.add_argument('--neuron.weight_decay', type=float, help='nucleus parameter weight decay.', default=0.25)
    parser.add_argument('--neuron.momentum', type=float, help='optimizer momentum.', default=0.8)
    parser.add_argument('--neuron.clip_gradients', type=float,
                        help='Implement gradient clipping to avoid exploding loss on smaller architectures.',
                        default=1.0)
    parser.add_argument('--neuron.batch_size_train', type=int, help='Training batch size.', default=32)
    parser.add_argument('--neuron.expert', type=str, help='HuggingFace model name for expert.',
                        default='benjamin/gerpt2-large')
    parser.add_argument('--neuron.device', type=str, help='Torch device for training.', default='cuda:0')
    parser.add_argument('--neuron.second_device', type=str, help='Torch second device training.',
                        default='cuda:0')
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


def tokenize(batch, tokenizer, tokenizer2):
    r"""
    Tokenizes a batch inputs sentences.
    Args:
        batch: Input batch of sentences.

    Returns:
        A batch of tokenized sequences.
    """
    new_batch = {}

    tokens = tokenizer(batch, return_offsets_mapping=True)
    new_batch['input_ids'] = tokens['input_ids']
    new_batch['offset_mapping'] = tokens['offset_mapping']

    tokens = tokenizer2(batch, return_offsets_mapping=True)
    new_batch['input_ids2'] = tokens['input_ids']
    new_batch['offset_mapping2'] = tokens['offset_mapping']

    return new_batch


def chunk(batch, block_size: int):
    r"""
    Concatenates and chunks a batch of token sequences into batches of length block_size.
    Args:
        batch: Input batch of tokenized sequences.
        block_size: Length of each token sequence in the batch.

    Returns:
        A new modified batch of shape [new_batch_size, block_size].
    """
    concat = {key: sum(batch[key], []) for key in ['input_ids', 'input_ids2']}

    for key in ['offset_mapping', 'offset_mapping2']:
        concat[key] = batch[key][0]  # first sequence

    for i in range(1, len(batch['offset_mapping'])):  # concat the offset_mappings over batch sequences
        for key in ['offset_mapping', 'offset_mapping2']:
            offset = concat[key][-1][-1]  # prev seq, last tuple, last index
            concat[key] += [(a + offset, b + offset) for a, b in batch[key][i]]  # append next seq offsets

    new_batch = {'input_ids': [], 'input_ids2': [],
                 'offset_mapping': [], 'offset_mapping2': []}
    i = block_size  # token index for offset_mapping
    j = block_size  # token index for offset_mapping2
    seg = concat['offset_mapping']  # segmentation
    seg2 = concat['offset_mapping2']  # segmentation2

    while i + 1 < len(seg) and j + 1 < len(seg2):
        x = seg[i][1]  # end pos over input
        y = seg2[j][1]  # end pos over input
        if x == y:  # aligned segmentation over input
            for k, key in [(i, 'input_ids'), (j, 'input_ids2')]:
                new_batch[key] += [concat[key][k - block_size:k + 2]]  # include extra validation token after alignment
            for k, key in [(i, 'offset_mapping'), (j, 'offset_mapping2')]:
                new_batch[key] += [
                    [_[0] for _ in concat[key][k - block_size:k + 2]]]  # include extra validation token after alignment
                # if len(new_batch['offset_mapping']) == 10 and len(new_batch['offset_mapping2']) == 10:
                #     print(i, j, x, y, len(new_batch['offset_mapping'][9]), len(new_batch['offset_mapping2'][9]),
                #           new_batch['offset_mapping'][9][-1], new_batch['offset_mapping2'][9][-1],
                #           concat['offset_mapping'][i], concat['offset_mapping2'][j])
            i += block_size + 2
            j += block_size + 2
        elif x < y:  # not yet aligned
            i += 1  # advance i
        else:  # not yet aligned
            j += 1  # advance j

    return new_batch


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
    dataset = load_dataset(config.dataset.name, config.dataset.subset, split=config.dataset.split)

    # Tokenize the dataset text sequences.
    # tokenizer = bittensor.tokenizer()
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2', local_files_only=False)
    tokenizer2 = AutoTokenizer.from_pretrained(config.neuron.expert, local_files_only=False)  # expert tokenizer

    dataset = dataset.map(lambda _batch: tokenize(_batch['text'], tokenizer, tokenizer2),
                          remove_columns=['text', 'title'],
                          batched=True, num_proc=config.dataset.num_workers)

    # Chunk the token sequences into fixed block_size length.
    dataset = dataset.map(lambda _batch: chunk(_batch, block_size),
                          batched=True, batch_size=10, num_proc=config.dataset.num_workers)  #

    # Format our dataset to outputs torch.Tensor to train a pytorch model.
    columns = ['input_ids', 'input_ids2',
               'offset_mapping', 'offset_mapping2']
    dataset.set_format(type='torch', columns=columns)

    distil_device = config.neuron.device

    # Choose teacher models with significantly different capacity to observe learning of
    #  significantly different peer weights for these by the nucleus distillation model.
    teachers = [
        {'name': config.neuron.expert,  # benjamin/gerpt2-large
         'dim': 1280,  # 36-layer, 1280-hidden, 20-heads, 774M parameters
         'device': config.neuron.device}
    ]

    for teacher in teachers:
        # Load pretrained teacher models with language-modeling heads
        teacher['model'] = AutoModelForCausalLM.from_pretrained(teacher['name']).half().to(teacher['device'])

    for teacher in teachers:
        # Adapt the teacher hidden dimension to the bittensor network dimension of bittensor by using
        #  a fully-connected layer to convert teacher hidden features to a size of bittensor.__network_dim__
        teacher['adaptor'] = nn.Linear(teacher['dim'], bittensor.__network_dim__, bias=False).to(distil_device)
        teacher['adaptor2'] = nn.Linear(teacher['dim'], bittensor.__network_dim__, bias=False).to(distil_device)
        teacher['adaptor3'] = nn.Linear(teacher['dim'], bittensor.__network_dim__, bias=False).to(distil_device)

    # Learn the dimension adaptors for the input teachers getting mixed
    adaptor_params = sum((list(teacher['adaptor'].parameters()) for teacher in teachers), [])
    adaptor_optimizer = torch.optim.AdamW(adaptor_params, lr=config.neuron.learning_rate)

    adaptor2_params = sum((list(teacher['adaptor2'].parameters()) for teacher in teachers), [])
    adaptor2_optimizer = torch.optim.AdamW(adaptor2_params, lr=config.neuron.learning_rate)

    adaptor3_params = sum((list(teacher['adaptor3'].parameters()) for teacher in teachers), [])
    adaptor3_optimizer = torch.optim.AdamW(adaptor3_params, lr=config.neuron.learning_rate)

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

    # Initialize another nucleus that distils from an expert, but applies masking correction distil-c(orrected)
    distilc_device = config.neuron.device
    distilc_config = config.copy()
    distilc_config.neuron.device = distilc_device
    distilc_config.nucleus.alpha_clm = 1.
    distilc_config.nucleus.alpha_clm_dis = 0.0
    distilc_config.nucleus.alpha_clm_rmt = 1.0
    distilc_config.nucleus.alpha_mse = 0.0
    distilc_config.nucleus.alpha_mse_hid = 1.0
    distilc_config.nucleus.alpha_ce = 0.0
    distilc_config.nucleus.alpha_cos = 0.0
    distilc_model = Nucleus4(distilc_config)
    distilc_model.peer_weights = nn.Parameter(torch.ones([len(teachers)], requires_grad=True, device=distilc_device))
    # Load same initialization as distil_model
    distilc_model.load_state_dict(distil_state, strict=True)
    distilc_model = distilc_model.to(distilc_device)

    print(distilc_model)
    print('distilc', distilc_model.alpha_ce, distilc_model.alpha_clm, distilc_model.alpha_clm_dis,
          distilc_model.alpha_clm_rmt, distilc_model.alpha_mse, distilc_model.alpha_mse_hid,
          distilc_model.alpha_cos)

    # Initialize another nucleus that distils from an expert and uses the same tokenizer: distil-t(okenizer)
    distilt_device = config.neuron.device
    distilt_config = config.copy()
    distilt_config.neuron.device = distilt_device
    distilt_config.nucleus.alpha_clm = 1.
    distilt_config.nucleus.alpha_clm_dis = 0.0
    distilt_config.nucleus.alpha_clm_rmt = 1.0
    distilt_config.nucleus.alpha_mse = 0.0
    distilt_config.nucleus.alpha_mse_hid = 1.0
    distilt_config.nucleus.alpha_ce = 0.0
    distilt_config.nucleus.alpha_cos = 0.0
    distilt_model = Nucleus(distilt_config)
    distilt_model.peer_weights = nn.Parameter(torch.ones([len(teachers)], requires_grad=True, device=distilt_device))
    # Load same initialization as distil_model
    distilt_model.load_state_dict(distil_state, strict=True)
    distilt_model = distilt_model.to(distilt_device)

    print(distilt_model)
    print('distilc', distilt_model.alpha_ce, distilt_model.alpha_clm, distilt_model.alpha_clm_dis,
          distilt_model.alpha_clm_rmt, distilt_model.alpha_mse, distilt_model.alpha_mse_hid,
          distilt_model.alpha_cos)

    # Initialize another nucleus that learns an lm head but without distillation
    undistilt_device = config.neuron.device
    undistilt_config = config.copy()
    undistilt_config.neuron.device = undistilt_device
    undistilt_config.nucleus.alpha_clm = 1.
    undistilt_config.nucleus.alpha_clm_dis = 0.0
    undistilt_config.nucleus.alpha_clm_rmt = 0.0
    undistilt_config.nucleus.alpha_mse = 0.0
    undistilt_config.nucleus.alpha_mse_hid = 0.0
    undistilt_config.nucleus.alpha_ce = 0.0
    undistilt_config.nucleus.alpha_cos = 0.0
    undistilt_model = Nucleus(undistilt_config)
    # undistil model won't distil, but need to create same-size parameter to load same initialization
    undistilt_model.peer_weights = nn.Parameter(torch.ones([len(teachers)], requires_grad=True, device=distil_device))
    # Load same initialization as distil_model
    undistilt_model.load_state_dict(distil_state, strict=True)
    undistilt_model = undistilt_model.to(undistilt_device)

    print(undistilt_model)
    print('undistilt', undistilt_model.alpha_ce, undistilt_model.alpha_clm, undistilt_model.alpha_clm_dis,
          undistilt_model.alpha_clm_rmt, undistilt_model.alpha_mse, undistilt_model.alpha_mse_hid,
          undistilt_model.alpha_cos)

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
    distilc_optimizer = torch.optim.AdamW(distilc_model.parameters(), lr=config.neuron.learning_rate)
    distilt_optimizer = torch.optim.AdamW(distilt_model.parameters(), lr=config.neuron.learning_rate)
    undistilt_optimizer = torch.optim.AdamW(undistilt_model.parameters(), lr=config.neuron.learning_rate)
    undistil_optimizer = torch.optim.AdamW(undistil_model.parameters(), lr=config.neuron.learning_rate)

    # Define learning rate scheduler (multiplier) for optimizer
    distil_scheduler = None
    distilc_scheduler = None
    distilt_scheduler = None
    undistilt_scheduler = None
    undistil_scheduler = None
    adaptor_scheduler = None

    if config.neuron.lr_scheduler == 'get_cosine_schedule_with_warmup':
        adaptor_scheduler = transformers.get_cosine_schedule_with_warmup(optimizer=adaptor_optimizer,
                                                                         num_warmup_steps=config.neuron.num_warmup_steps,
                                                                         num_training_steps=config.neuron.n_epochs)
        distil_scheduler = transformers.get_cosine_schedule_with_warmup(optimizer=distil_optimizer,
                                                                        num_warmup_steps=config.neuron.num_warmup_steps,
                                                                        num_training_steps=config.neuron.n_epochs)
        distilc_scheduler = transformers.get_cosine_schedule_with_warmup(optimizer=distilc_optimizer,
                                                                         num_warmup_steps=config.neuron.num_warmup_steps,
                                                                         num_training_steps=config.neuron.n_epochs)
        distilt_scheduler = transformers.get_cosine_schedule_with_warmup(optimizer=distilt_optimizer,
                                                                         num_warmup_steps=config.neuron.num_warmup_steps,
                                                                         num_training_steps=config.neuron.n_epochs)
        undistilt_scheduler = transformers.get_cosine_schedule_with_warmup(optimizer=undistilt_optimizer,
                                                                           num_warmup_steps=config.neuron.num_warmup_steps,
                                                                           num_training_steps=config.neuron.n_epochs)
        undistil_scheduler = transformers.get_cosine_schedule_with_warmup(optimizer=undistil_optimizer,
                                                                          num_warmup_steps=config.neuron.num_warmup_steps,
                                                                          num_training_steps=config.neuron.n_epochs)

    if config.neuron.use_wandb:
        bittensor.wandb(config)  # Initialize wandb logging
        wandb.watch(distil_model)  # Track model parameters and gradients
        wandb.watch(distilc_model)  # Track model parameters and gradients
        wandb.watch(distilt_model)  # Track model parameters and gradients
        wandb.watch(undistilt_model)  # Track model parameters and gradients
        wandb.watch(undistil_model)  # Track model parameters and gradients
        for teacher in teachers:
            wandb.watch(teacher['adaptor'])
            wandb.watch(teacher['adaptor2'])
            wandb.watch(teacher['adaptor3'])
        wandb_table_data = []

    for dataset_epoch in range(3):
        # Define pytorch dataloader with shuffled batches of batch_size token sequences of block_size length.
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch, batch in enumerate(dataloader):
            with torch.no_grad():
                # for col in columns:
                #     print(col, len(batch[col]), len(batch[col][0]), type(batch[col]))
                input_ids2 = batch['input_ids2'].to(distil_device)  # tokenized by expert tokenizer
                target2 = input_ids2[:, -1]  # held out target of last token
                input_ids2 = input_ids2[:, :-1]  # entire sequence except last token

                teacher_inputs = {}

                for teacher in teachers:
                    if teacher['device'] not in teacher_inputs:
                        teacher_inputs[teacher['device']] = input_ids2.clone().to(teacher['device'])

                    teacher_input_ids = teacher_inputs[teacher['device']]
                    teacher_output = teacher['model'](input_ids=teacher_input_ids, output_hidden_states=True)
                    teacher['hidden_states'] = teacher_output.hidden_states[-1].float()

                    shift_logits = teacher_output.logits[..., :-1, :].float().contiguous()
                    shift_labels = teacher_input_ids[..., 1:].contiguous()
                    teacher['loss_clm'] = nn.CrossEntropyLoss()(shift_logits.view(-1, shift_logits.size(-1)),
                                                                shift_labels.view(-1))
                    predictions = shift_logits.detach().max(2).indices
                    teacher['acc'] = (predictions == shift_labels).sum().item() / predictions.nelement()

                    teacher['prediction'] = teacher_output.logits[:, -1, :].argmax(-1)  # predict unseen last token
                    teacher['predictions'] = tokenizer.decode(teacher_output.logits[0].argmax(-1).detach())

                    teacher_target = target2.clone().to(teacher['device'])
                    teacher['target_acc'] = (teacher['prediction'] == teacher_target).sum().item() / len(teacher_target)

                adaptor_lr = adaptor_optimizer.param_groups[0]['lr']

                input_ids = batch['input_ids'].to(distil_device)
                target = input_ids[:, -1]  # held out target of last token
                input_ids = input_ids[:, :-1]  # entire sequence except last token

            teacher = teachers[0]
            distil_teacher_inputs = teacher['adaptor'](
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

                distilc_input_ids = input_ids.detach().to(distilc_device)

            teacher = teachers[0]
            distilc_teacher_inputs = teacher['adaptor2'](
                teacher['hidden_states'].detach().to(distilc_device))

            distilc_output = distilc_model.remote_forward(distilc_input_ids, training=True,
                                                          teacher_inputs=distilc_teacher_inputs,
                                                          offset_mapping=batch['offset_mapping'],
                                                          offset_mapping2=batch[
                                                              'offset_mapping2'])  # forward pass in local transformer model
            distilc_total_loss = (distilc_model.alpha_clm * distilc_output.loss_clm +
                                  distilc_model.alpha_clm_dis * distilc_output.loss_clm_dis +
                                  distilc_model.alpha_clm_rmt * distilc_output.loss_clm_rmt +
                                  distilc_model.alpha_mse * distilc_output.loss_mse +
                                  distilc_model.alpha_mse_hid * distilc_output.loss_mse_hid +
                                  distilc_model.alpha_ce * distilc_output.loss_ce +
                                  distilc_model.alpha_cos * distilc_output.loss_cos)

            with torch.no_grad():
                distilc_loss_clm = distilc_output.loss_clm
                distilc_loss_clm_dis = distilc_output.loss_clm_dis
                distilc_loss_clm_rmt = distilc_output.loss_clm_rmt
                distilc_loss_mse = distilc_output.loss_mse
                distilc_loss_mse_hid = distilc_output.loss_mse_hid
                distilc_loss_ce = distilc_output.loss_ce
                distilc_loss_cos = distilc_output.loss_cos
                distilc_acc = distilc_output.local_accuracy  # training accuracy on next token prediction in train sequence with masking
                distilc_remote_acc = distilc_output.remote_accuracy  # training accuracy on next token prediction in train sequence with masking
                distilc_lr = distilc_optimizer.param_groups[0]['lr']  # record actual learning rate
                # distilc_weight_lr = distilc_weight_optimizer.param_groups[0]['lr']  # record actual learning rate

                distilc_prediction = distilc_output.local_target[:, -1, :].argmax(-1)  # predict unseen last token
                distilc_target_acc = (distilc_prediction == target).sum().item() / len(
                    target)  # validation accuracy on predicting unseen token

                distilc_remote_prediction = distilc_output.remote_target[:, -1, :].argmax(
                    -1)  # predict unseen last token
                distilc_remote_target_acc = (distilc_remote_prediction == target).sum().item() / len(
                    target)  # validation accuracy on predicting unseen token

                distilt_input_ids = input_ids2.detach().to(distilt_device)  # use same tokenizer as teacher

            teacher = teachers[0]
            distilt_teacher_inputs = teacher['adaptor3'](
                teacher['hidden_states'].detach().to(distilt_device))

            distilt_output = distilt_model.remote_forward(distilt_input_ids, training=True,
                                                          teacher_inputs=distilt_teacher_inputs)  # forward pass in local transformer model
            distilt_total_loss = (distilt_model.alpha_clm * distilt_output.loss_clm +
                                  distilt_model.alpha_clm_dis * distilt_output.loss_clm_dis +
                                  distilt_model.alpha_clm_rmt * distilt_output.loss_clm_rmt +
                                  distilt_model.alpha_mse * distilt_output.loss_mse +
                                  distilt_model.alpha_mse_hid * distilt_output.loss_mse_hid +
                                  distilt_model.alpha_ce * distilt_output.loss_ce +
                                  distilt_model.alpha_cos * distilt_output.loss_cos)

            with torch.no_grad():
                distilt_loss_clm = distilt_output.loss_clm
                distilt_loss_clm_dis = distilt_output.loss_clm_dis
                distilt_loss_clm_rmt = distilt_output.loss_clm_rmt
                distilt_loss_mse = distilt_output.loss_mse
                distilt_loss_mse_hid = distilt_output.loss_mse_hid
                distilt_loss_ce = distilt_output.loss_ce
                distilt_loss_cos = distilt_output.loss_cos
                distilt_acc = distilt_output.local_accuracy  # training accuracy on next token prediction in train sequence with masking
                distilt_remote_acc = distilt_output.remote_accuracy  # training accuracy on next token prediction in train sequence with masking
                distilt_lr = distilt_optimizer.param_groups[0]['lr']  # record actual learning rate
                # distilt_weight_lr = distilt_weight_optimizer.param_groups[0]['lr']  # record actual learning rate

                distilt_prediction = distilt_output.local_target[:, -1, :].argmax(-1)  # predict unseen last token
                distilt_target_acc = (distilt_prediction == target2).sum().item() / len(
                    target2)  # validation accuracy on predicting unseen token

                distilt_remote_prediction = distilt_output.remote_target[:, -1, :].argmax(
                    -1)  # predict unseen last token
                distilt_remote_target_acc = (distilt_remote_prediction == target2).sum().item() / len(
                    target2)  # validation accuracy on predicting unseen token

                undistilt_input_ids = input_ids2.detach().to(undistilt_device)

            undistilt_output = undistilt_model.local_forward(undistilt_input_ids,
                                                             training=True)  # forward pass in local transformer model
            undistilt_loss = undistilt_output.loss_clm

            with torch.no_grad():
                undistilt_acc = undistilt_output.local_accuracy  # training accuracy on next token prediction in train sequence with masking
                undistilt_lr = undistilt_optimizer.param_groups[0]['lr']  # record actual learning rate
                undistilt_prediction = undistilt_output.local_target[:, -1, :].argmax(-1)  # predict unseen last token
                undistilt_target = target2.to(undistilt_device)
                undistilt_target_acc = (undistilt_prediction == undistilt_target).sum().item() / len(
                    undistilt_target)  # validation accuracy on predicting unseen token

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
                                               distilc_total_loss.item(),
                                               distil_acc, distilc_acc,
                                               distil_target_acc, distil_remote_target_acc,
                                               teachers[-1]['target_acc'], distilc_target_acc,
                                               distil_lr), end=' ')

                if epoch % 1000 == 0:
                    input_decoded = tokenizer.decode(input_ids[0])
                    distil_predictions = distil_output.local_target[0].detach().argmax(-1)
                    distilc_predictions = distilc_output.local_target[0].detach().argmax(-1)

                    print('\n.\n', input_decoded, '\n...\n')
                    print(list(zip([tokenizer.decode(_) for _ in input_ids[0]],
                                   [tokenizer.decode(_) for _ in distil_predictions])), '\n.\n')

                    distil_predictions = tokenizer.decode(distil_predictions)
                    distilc_predictions = tokenizer.decode(distilc_predictions)
                    if config.neuron.use_wandb:
                        wandb_table_data += [[epoch,
                                              distil_target_acc,
                                              distil_predictions, distilc_predictions, input_decoded] +
                                             [teacher['predictions'] for teacher in teachers]]

                if config.neuron.use_wandb:
                    if epoch % 5000 == 0:
                        wandb_table = wandb.Table(columns=['epoch',
                                                           'distil_target_acc',
                                                           'distil_predictions', 'distilc_predictions', 'input'] +
                                                          ['%s' % teacher['name'] for teacher in teachers])
                        for row in wandb_table_data:
                            wandb_table.add_data(*row)

                        torch.save(distil_model.state_dict(),
                                   "{}/distil_model_{}.torch".format(config.wandb.directory, config.wandb.name))

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

                                 'distilc_loss_clm': distilc_loss_clm.item(),
                                 'distilc_loss_clm_dis': distilc_loss_clm_dis.item(),
                                 'distilc_loss_clm_rmt': distilc_loss_clm_rmt.item(),
                                 'distilc_loss_mse': distilc_loss_mse.item(),
                                 'distilc_loss_mse_hid': distilc_loss_mse_hid.item(),
                                 'distilc_loss_ce': distilc_loss_ce.item(),
                                 'distilc_loss_cos': distilc_loss_cos.item(),
                                 'distilc_total_loss': distilc_total_loss.item(),
                                 'distilc_acc': distilc_acc,
                                 'distilc_remote_acc': distilc_remote_acc,
                                 'distilc_target_acc': distilc_target_acc,
                                 'distilc_remote_target_acc': distilc_remote_target_acc,
                                 'distilc_lr': distilc_lr,
                                 'distilc_expert_calls': len(teachers) * batch_size * epoch,

                                 'distilt_loss_clm': distilt_loss_clm.item(),
                                 'distilt_loss_clm_dis': distilt_loss_clm_dis.item(),
                                 'distilt_loss_clm_rmt': distilt_loss_clm_rmt.item(),
                                 'distilt_loss_mse': distilt_loss_mse.item(),
                                 'distilt_loss_mse_hid': distilt_loss_mse_hid.item(),
                                 'distilt_loss_ce': distilt_loss_ce.item(),
                                 'distilt_loss_cos': distilt_loss_cos.item(),
                                 'distilt_total_loss': distilt_total_loss.item(),
                                 'distilt_acc': distilt_acc,
                                 'distilt_remote_acc': distilt_remote_acc,
                                 'distilt_target_acc': distilt_target_acc,
                                 'distilt_remote_target_acc': distilt_remote_target_acc,
                                 'distilt_lr': distilt_lr,
                                 'distilt_expert_calls': len(teachers) * batch_size * epoch,

                                 'undistilt_loss': undistilt_loss.item(),
                                 'undistilt_acc': undistilt_acc,
                                 'undistilt_target_acc': undistilt_target_acc,
                                 'undistilt_lr': undistilt_lr,

                                 'undistil_loss': undistil_loss.item(),
                                 'undistil_acc': undistil_acc,
                                 'undistil_target_acc': undistil_target_acc,
                                 'undistil_lr': undistil_lr,

                                 'adaptor_lr': adaptor_lr,
                                 # 'experts_per_input': experts_per_input,
                                 'batch': epoch}

                    wandb_log = [wandb_log] + [{'teacher%d_loss_clm' % i: teacher['loss_clm'].item(),
                                                'teacher%d_acc' % i: teacher['acc'],
                                                'teacher%d_target_acc' % i: teacher['target_acc'],
                                                } for i, teacher in enumerate(teachers)]

                    wandb.log({k: v for d in wandb_log for k, v in d.items()})

            torch.cuda.empty_cache()

            distil_total_loss.backward()  # accumulate gradients wrt training loss
            distilc_total_loss.backward()  # accumulate gradients wrt training loss
            distilt_total_loss.backward()  # accumulate gradients wrt training loss
            undistilt_loss.backward()  # accumulate gradients wrt training loss
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

                torch.nn.utils.clip_grad_norm_(distilc_model.parameters(), 0.5)
                distilc_optimizer.step()  # update model parameters to reduce loss
                distilc_optimizer.zero_grad()  # remove previously accumulated gradients
                if distilc_scheduler:
                    distilc_scheduler.step()  # update learning rate multiplier

                torch.nn.utils.clip_grad_norm_(distilt_model.parameters(), 0.5)
                distilt_optimizer.step()  # update model parameters to reduce loss
                distilt_optimizer.zero_grad()  # remove previously accumulated gradients
                if distilt_scheduler:
                    distilt_scheduler.step()  # update learning rate multiplier

                torch.nn.utils.clip_grad_norm_(undistilt_model.parameters(), 0.5)
                undistilt_optimizer.step()  # update model parameters to reduce loss
                undistilt_optimizer.zero_grad()  # remove previously accumulated gradients
                if undistilt_scheduler:
                    undistilt_scheduler.step()  # update learning rate multiplier

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
                adaptor_scheduler.step()

            torch.cuda.empty_cache()


if __name__ == '__main__':
    use_config = main_config()
    main(use_config)
