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

""" Model learning test for template_miner local_forward stage.
"""

import wandb
import torch
import argparse
import bittensor
import transformers

from datasets import load_dataset
from torch.utils.data import DataLoader
from bittensor._neuron.text.template_miner.nucleus_impl import Nucleus


def modify_args(parser: argparse.ArgumentParser):
    r""" Modify custom params in the parser for this test.
    """
    parser.add_argument('--wandb.name', type=str, help='''Optionally pass wandb run name for use_wandb''', default='template_miner_local')
    parser.add_argument('--wandb.project', type=str, help='''Optionally pass wandb project name for use_wandb''', default='neuron-tests')
    parser.add_argument('--wandb.tags', type=str, help='''Optionally pass wandb tags for use_wandb''', default='neuron, test, template_miner_local')
    parser.add_argument('--wandb.run_group', type=str, help='''Optionally pass wandb group name for use_wandb''', default='template_miner_local')

    parser.add_argument('--dataset.batch_size', type=int, help='Batch size.', default=64)
    parser.add_argument('--dataset.block_size', type=int, help='Number of text items to pull for each example..', default=50)
    parser.add_argument('--dataset.num_workers',  type=int, help='Number of workers for data loader.', default=16)
    parser.add_argument('--dataset.name', type=str, help='Which dataset to use.', default='bookcorpusopen')
    parser.add_argument('--dataset.split', type=str, help='Which split to use (train/test/validation).', default='train')

    parser.add_argument('--nucleus.nhid', type=int, help='the dimension of the feedforward network model in nn.TransformerEncoder', default=768)
    parser.add_argument('--nucleus.nhead', type=int, help='the number of heads in the multihead attention models', default=8)
    parser.add_argument('--nucleus.nlayers', type=int, help='the number of nn.TransformerEncoderLayer in nn.TransformerEncoder', default=8)
    parser.add_argument('--nucleus.dropout', type=float, help='the dropout value', default=0.1)

    parser.add_argument('--neuron.learning_rate', type=float, help='Training initial learning rate.', default=1e-4)
    parser.add_argument('--neuron.weight_decay', type=float, help='nucleus parameter weight decay.', default=0.25)
    parser.add_argument('--neuron.momentum', type=float, help='optimizer momentum.', default=0.8)
    parser.add_argument('--neuron.clip_gradients', type=float, help='Implement gradient clipping to avoid exploding loss on smaller architectures.', default=1.0)
    parser.add_argument('--neuron.batch_size_train', type=int, help='Training batch size.', default=32)
    parser.add_argument('--neuron.device', type=str, help='Torch device for training.', default='cuda:0')
    parser.add_argument('--neuron.use_wandb', action='store_true', help='''neuron activates its weights and biases powers''', default=False)
    parser.add_argument('--neuron.n_epochs', type=int, help='Number of training epochs.', default=300000)
    parser.add_argument('--neuron.lr_scheduler', type=str, help='Learning rate scheduler name.', default='get_cosine_with_hard_restarts_schedule_with_warmup')
    parser.add_argument('--neuron.num_warmup_steps', type=int, help='Learning rate scheduler number of warmup steps.', default=60000)
    parser.add_argument('--neuron.num_cycles', type=int, help='Learning rate scheduler number of cycles for hard restart.', default=5)


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
        key: [val[i:i+block_size] for i in range(0, trunc_length, block_size)] for key, val in concatenated.items()
    }
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
    dataset = load_dataset(config.dataset.name, split=config.dataset.split)

    # Tokenize the dataset text sequences.
    tokenizer = bittensor.tokenizer()
    dataset = dataset.map(lambda batch: tokenizer(batch['text']), remove_columns=['text', 'title'],
                          batched=True, num_proc=config.dataset.num_workers)

    # Chunk the token sequences into fixed block_size length.
    dataset = dataset.map(lambda batch: chunk(batch, block_size),
                          batched=True, batch_size=2, num_proc=config.dataset.num_workers)  #

    # Format our dataset to outputs torch.Tensor to train a pytorch model.
    columns = ['input_ids', 'attention_mask']
    dataset.set_format(type='torch', columns=columns)

    # Define pytorch dataloader with shuffled batches of batch_size token sequences of block_size length.
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    device = config.neuron.device

    # Initialize nucleus pytorch model and move to specified device
    model = Nucleus(config).to(device)

    # Define optimizer over all model parameters at specified learning rate
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.neuron.learning_rate)

    # Define learning rate scheduler (multiplier) for optimizer
    scheduler = None
    if config.neuron.lr_scheduler == 'get_cosine_schedule_with_warmup':
        scheduler = transformers.get_cosine_schedule_with_warmup(optimizer=optimizer,
                                                                 num_warmup_steps=config.neuron.num_warmup_steps,
                                                                 num_training_steps=config.neuron.n_epochs)
    elif config.neuron.lr_scheduler == 'get_cosine_with_hard_restarts_schedule_with_warmup':
        scheduler = transformers.get_cosine_with_hard_restarts_schedule_with_warmup(optimizer=optimizer,
                                                                                    num_warmup_steps=config.neuron.num_warmup_steps,
                                                                                    num_training_steps=config.neuron.n_epochs,
                                                                                    num_cycles=config.neuron.num_cycles)

    if config.neuron.use_wandb:
        bittensor.wandb(config)  # Initialize wandb logging
        wandb.watch(model)  # Track model parameters and gradients
        wandb_table_data = []

    for epoch, batch in enumerate(dataloader):
        input_ids = batch['input_ids'].to(device)
        target = input_ids[:, -1]  # held out target of last token
        input_ids = input_ids[:, :-1]  # entire sequence except last token

        output = model.local_forward(input_ids, training=True)  # forward pass in local transformer model
        acc = output.local_accuracy  # training accuracy on next token prediction in train sequence with masking
        loss = output.local_target_loss
        lr = optimizer.param_groups[0]['lr']  # record actual learning rate

        prediction = output.local_target[:, -1, :].argmax(-1)  # predict unseen last token
        target_acc = (prediction == target).sum().item() / len(target)  # validation accuracy on predicting unseen token

        if epoch % 100 == 0:
            print('%d: %.1f (%.2f, %.2f, %f)' % (epoch, loss.item(), acc, target_acc, lr), end=' ')

        optimizer.zero_grad()  # remove previously accumulated gradients
        loss.backward()  # accumulate gradients wrt training loss
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()  # update model parameters to reduce loss
        if scheduler:
            scheduler.step()  # update learning rate multiplier

        if epoch % 1000 == 0:
            predictions = output.local_target.argmax(-1)  # predict next token in sequence
            if config.neuron.use_wandb:
                wandb_table_data += [[epoch, loss, acc, target_acc, lr,
                                      tokenizer.decode(predictions[0]), tokenizer.decode(input_ids[0])]]

            print('\n.\n', tokenizer.decode(input_ids[0]), '\n...\n')
            print(list(zip([tokenizer.decode(_) for _ in input_ids[0]],
                           [tokenizer.decode(_) for _ in predictions[0]])), '\n.\n')

        if config.neuron.use_wandb:
            wandb.log({'loss': loss,
                       'acc': acc,
                       'target_acc': target_acc,
                       'lr': lr})

            if epoch % 10000 == 0:
                wandb_table = wandb.Table(columns=['epoch', 'loss', 'acc', 'target_acc', 'lr', 'predict', 'input'])
                for row in wandb_table_data:
                    wandb_table.add_data(*row)
                wandb.log({'training_samples': wandb_table})


if __name__ == '__main__':
    use_config = main_config()
    main(use_config)
