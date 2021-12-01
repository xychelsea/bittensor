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

""" Distillation learning test for template_miner.
"""

import wandb
import torch
import argparse
import bittensor
import transformers

from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import GPT2TokenizerFast, GPT2LMHeadModel
# from bittensor._neuron.text.template_miner.nucleus_impl import Nucleus
from nucleus_295_3 import Nucleus


def modify_args(parser: argparse.ArgumentParser):
    r""" Modify custom params in the parser for this test.
    """
    parser.add_argument('--wandb.name', type=str, help='''Optionally pass wandb run name for use_wandb''',
                        default='BIT-295-distil-3')
    parser.add_argument('--wandb.project', type=str, help='''Optionally pass wandb project name for use_wandb''',
                        default='neuron-tests')
    parser.add_argument('--wandb.tags', type=str, help='''Optionally pass wandb tags for use_wandb''',
                        default='hf losses, no-pos-enc, neuron, test, template_miner_distil')
    parser.add_argument('--wandb.run_group', type=str, help='''Optionally pass wandb group name for use_wandb''',
                        default='template_miner_distil')

    parser.add_argument('--dataset.batch_size', type=int, help='Batch size.', default=16)
    parser.add_argument('--dataset.block_size', type=int, help='Number of text items to pull for each example..',
                        default=80)
    parser.add_argument('--dataset.num_workers', type=int, help='Number of workers for data loader.', default=16)
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
    parser.add_argument("--nucleus.alpha_mse", default=0.0, type=float,
                        help="Linear weight of the MSE loss. Must be >=0.")
    parser.add_argument(
        "--nucleus.alpha_cos", default=0.0, type=float, help="Linear weight of the cosine embedding loss. Must be >=0."
    )

    parser.add_argument('--neuron.learning_rate', type=float, help='Training initial learning rate.', default=5e-4)
    parser.add_argument('--neuron.weight_decay', type=float, help='nucleus parameter weight decay.', default=0.25)
    parser.add_argument('--neuron.momentum', type=float, help='optimizer momentum.', default=0.8)
    parser.add_argument('--neuron.clip_gradients', type=float,
                        help='Implement gradient clipping to avoid exploding loss on smaller architectures.',
                        default=1.0)
    parser.add_argument('--neuron.batch_size_train', type=int, help='Training batch size.', default=16)
    parser.add_argument('--neuron.device', type=str, help='Torch device for training.', default='cuda:0')
    parser.add_argument('--neuron.second_device', type=str, help='Torch device for second distillation model.',
                        default='cuda:1')
    parser.add_argument('--neuron.teacher_device', type=str, help='Torch device for teacher model training.',
                        default='cuda:2')
    parser.add_argument('--neuron.use_wandb', action='store_true',
                        help='''neuron activates its weights and biases powers''', default=False)
    parser.add_argument('--neuron.n_epochs', type=int, help='Number of training epochs.', default=300000)
    parser.add_argument('--neuron.lr_scheduler', type=str, help='Learning rate scheduler name.',
                        default='get_cosine_with_hard_restarts_schedule_with_warmup')
    parser.add_argument('--neuron.num_warmup_steps', type=int, help='Learning rate scheduler number of warmup steps.',
                        default=20000)
    parser.add_argument('--neuron.num_cycles', type=int,
                        help='Learning rate scheduler number of cycles for hard restart.', default=5)


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
    teacher_model_name = 'gpt2-medium'
    tokenizer = GPT2TokenizerFast.from_pretrained(teacher_model_name, local_files_only=False)
    dataset = dataset.map(lambda _batch: tokenizer(_batch['text']), remove_columns=['text', 'title'],
                          batched=True, num_proc=config.dataset.num_workers)

    # Chunk the token sequences into fixed block_size length.
    dataset = dataset.map(lambda _batch: chunk(_batch, block_size),
                          batched=True, batch_size=2, num_proc=config.dataset.num_workers)  #

    # Format our dataset to outputs torch.Tensor to train a pytorch model.
    columns = ['input_ids', 'attention_mask']
    dataset.set_format(type='torch', columns=columns)

    # Define pytorch dataloader with shuffled batches of batch_size token sequences of block_size length.
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize nucleus pytorch model to perform distillation from teacher and move to specified device
    distil_device = config.neuron.device
    distil_config = config.copy()
    distil_config.neuron.device = distil_device
    distil_model = Nucleus(distil_config).to(distil_device)
    # Save model to capture unique parameter initialization for reuse in distil2 model.
    distil_state = distil_model.state_dict()

    # Initialize another nucleus that performs distillation
    distil2_device = config.neuron.second_device
    distil2_config = config.copy()
    distil2_config.neuron.device = distil2_device
    distil2_config.nucleus.alpha_cos = 0.5
    distil2_model = Nucleus(distil2_config)
    # Load same initialization as distil_model
    distil2_model.load_state_dict(distil_state, strict=True)
    distil2_model = distil2_model.to(distil2_device)

    # Initialize another nucleus that plugs random features directly instead of distillation
    random_device = config.neuron.teacher_device
    random_config = config.copy()
    random_config.neuron.device = random_device
    random_config.nucleus.nlayers = 1
    random_model = Nucleus(random_config)
    # Load same initialization as distil_model
    random_model.load_state_dict(distil_state, strict=False)
    random_model = random_model.to(random_device)

    # Load pretrained teacher model with language-modeling head
    teacher_device = config.neuron.teacher_device
    teacher_model = GPT2LMHeadModel.from_pretrained(teacher_model_name).to(teacher_device)

    # Define optimizer over all model parameters at specified learning rate
    distil_optimizer = torch.optim.AdamW(distil_model.parameters(), lr=config.neuron.learning_rate)
    distil2_optimizer = torch.optim.AdamW(distil2_model.parameters(), lr=config.neuron.learning_rate)
    random_optimizer = torch.optim.AdamW(random_model.parameters(), lr=config.neuron.learning_rate)

    # Define learning rate scheduler (multiplier) for optimizer
    distil_scheduler = None
    distil2_scheduler = None
    random_scheduler = None

    if config.neuron.lr_scheduler == 'get_cosine_schedule_with_warmup':
        distil_scheduler = transformers.get_cosine_schedule_with_warmup(optimizer=distil_optimizer,
                                                                        num_warmup_steps=config.neuron.num_warmup_steps,
                                                                        num_training_steps=config.neuron.n_epochs)
        distil2_scheduler = transformers.get_cosine_schedule_with_warmup(optimizer=distil2_optimizer,
                                                                         num_warmup_steps=config.neuron.num_warmup_steps,
                                                                         num_training_steps=config.neuron.n_epochs)
        random_scheduler = transformers.get_cosine_schedule_with_warmup(optimizer=random_optimizer,
                                                                        num_warmup_steps=config.neuron.num_warmup_steps,
                                                                        num_training_steps=config.neuron.n_epochs)

    elif config.neuron.lr_scheduler == 'get_cosine_with_hard_restarts_schedule_with_warmup':
        distil_scheduler = transformers.get_cosine_with_hard_restarts_schedule_with_warmup(optimizer=distil_optimizer,
                                                                                           num_warmup_steps=config.neuron.num_warmup_steps,
                                                                                           num_training_steps=config.neuron.n_epochs,
                                                                                           num_cycles=config.neuron.num_cycles)
        distil2_scheduler = transformers.get_cosine_with_hard_restarts_schedule_with_warmup(optimizer=distil2_optimizer,
                                                                                            num_warmup_steps=config.neuron.num_warmup_steps,
                                                                                            num_training_steps=config.neuron.n_epochs,
                                                                                            num_cycles=config.neuron.num_cycles)
        random_scheduler = transformers.get_cosine_with_hard_restarts_schedule_with_warmup(optimizer=random_optimizer,
                                                                                           num_warmup_steps=config.neuron.num_warmup_steps,
                                                                                           num_training_steps=config.neuron.n_epochs,
                                                                                           num_cycles=config.neuron.num_cycles)

    if config.neuron.use_wandb:
        bittensor.wandb(config)  # Initialize wandb logging
        wandb.watch(distil_model)  # Track model parameters and gradients
        wandb.watch(distil2_model)  # Track model parameters and gradients
        wandb.watch(random_model)  # Track model parameters and gradients
        wandb_table_data = []

    print(distil_model)
    print(random_model)

    for epoch, batch in enumerate(dataloader):
        with torch.no_grad():
            input_ids = batch['input_ids'].to(distil_device)
            target = input_ids[:, -1]  # held out target of last token
            input_ids = input_ids[:, :-1]  # entire sequence except last token

            teacher_input_ids = input_ids.clone().to(teacher_device)
            teacher_output = teacher_model(input_ids=teacher_input_ids, output_hidden_states=True)
            teacher_hidden_states = teacher_output.hidden_states[-1]
            teacher_prediction = teacher_output.logits[:, -1, :].argmax(-1)  # predict unseen last token
            teacher_predictions = tokenizer.decode(teacher_output.logits[0].argmax(-1).detach())

            teacher_target = target.clone().to(teacher_device)
            teacher_target_acc = (teacher_prediction == teacher_target).sum().item() / len(teacher_target)

            # Remote serves teacher model to be distilled
            distil_inputs = teacher_hidden_states.detach().to(distil_device)
            distil2_inputs = teacher_hidden_states.detach().to(distil2_device)

            distil2_input_ids = input_ids.to(distil2_device)

        distil_output = distil_model.remote_forward(input_ids, training=True,
                                                    teacher_inputs=distil_inputs)  # forward pass in local transformer model
        distil_total_loss = (distil_model.alpha_clm * distil_output.loss_clm +
                             distil_model.alpha_mse * distil_output.loss_mse +
                             distil_model.alpha_ce * distil_output.loss_ce +
                             distil_model.alpha_cos * distil_output.loss_cos +
                             distil_output.remote_target_loss)

        distil2_output = distil2_model.remote_forward(distil2_input_ids, training=True,
                                                      teacher_inputs=distil2_inputs)  # forward pass in local transformer model
        distil2_total_loss = (distil2_model.alpha_clm * distil2_output.loss_clm +
                              distil2_model.alpha_mse * distil2_output.loss_mse +
                              distil2_model.alpha_ce * distil2_output.loss_ce +
                              distil2_model.alpha_cos * distil2_output.loss_cos +
                              distil2_output.remote_target_loss)

        with torch.no_grad():
            distil_loss_clm = distil_output.loss_clm
            distil_loss_mse = distil_output.loss_mse
            distil_loss_ce = distil_output.loss_ce
            distil_loss_cos = distil_output.loss_cos
            distil_remote_target_loss = distil_output.remote_target_loss
            distil_acc = distil_output.local_accuracy  # training accuracy on next token prediction in train sequence with masking
            distil_lr = distil_optimizer.param_groups[0]['lr']  # record actual learning rate

            distil_prediction = distil_output.local_target[:, -1, :].argmax(-1)  # predict unseen last token
            distil_target_acc = (distil_prediction == target).sum().item() / len(
                target)  # validation accuracy on predicting unseen token

            distil_remote_prediction = distil_output.remote_target[:, -1, :].argmax(-1)  # predict unseen last token
            distil_remote_target_acc = (distil_remote_prediction == target).sum().item() / len(
                target)  # validation accuracy on predicting unseen token

            distil2_loss_clm = distil2_output.loss_clm
            distil2_loss_mse = distil2_output.loss_mse
            distil2_loss_ce = distil2_output.loss_ce
            distil2_loss_cos = distil2_output.loss_cos
            distil2_remote_target_loss = distil2_output.remote_target_loss
            distil2_acc = distil2_output.local_accuracy  # training accuracy on next token prediction in train sequence with masking
            distil2_lr = distil2_optimizer.param_groups[0]['lr']  # record actual learning rate

            distil2_target = target.to(distil2_device)
            distil2_prediction = distil2_output.local_target[:, -1, :].argmax(-1)  # predict unseen last token
            distil2_target_acc = (distil2_prediction == distil2_target).sum().item() / len(
                distil2_target)  # validation accuracy on predicting unseen token

            distil2_remote_prediction = distil2_output.remote_target[:, -1, :].argmax(-1)  # predict unseen last token
            distil2_remote_target_acc = (distil2_remote_prediction == distil2_target).sum().item() / len(
                distil2_target)  # validation accuracy on predicting unseen token

            random_input_ids = input_ids.detach().to(random_device)

        random_output = random_model.local_forward(random_input_ids,
                                                   training=True)  # forward pass in local transformer model
        random_loss = random_output.loss_clm

        with torch.no_grad():
            random_acc = random_output.local_accuracy  # training accuracy on next token prediction in train sequence with masking
            random_lr = random_optimizer.param_groups[0]['lr']  # record actual learning rate
            random_prediction = random_output.local_target[:, -1, :].argmax(-1)  # predict unseen last token
            random_target = target.to(random_device)
            random_target_acc = (random_prediction == random_target).sum().item() / len(
                random_target)  # validation accuracy on predicting unseen token

            if epoch % 100 == 0:
                print('%d: %.1f %.1f %.1f %.1f '
                      '(%.2f, %.2f, %.2f, %.2f, %.2f, '
                      '%.2f, %.2f, %f)' % (epoch, distil_total_loss.item(),
                                           distil2_total_loss.item(), random_loss.item(),
                                           distil_acc, distil2_acc, random_acc,
                                           distil_target_acc, distil_remote_target_acc, distil2_target_acc,
                                           teacher_target_acc, random_target_acc,
                                           distil_lr), end=' ')

            if epoch % 1000 == 0:
                input_decoded = tokenizer.decode(input_ids[0])
                distil_predictions = distil_output.local_target[0].detach().argmax(-1)
                distil2_predictions = distil2_output.local_target[0].detach().argmax(-1)

                print('\n.\n', input_decoded, '\n...\n')
                print(list(zip([tokenizer.decode(_) for _ in input_ids[0]],
                               [tokenizer.decode(_) for _ in distil_predictions])), '\n.\n')

                distil_predictions = tokenizer.decode(distil_predictions)
                distil2_predictions = tokenizer.decode(distil2_predictions)
                if config.neuron.use_wandb:
                    wandb_table_data += [[epoch,
                                          distil_target_acc,
                                          distil_predictions, distil2_predictions, teacher_predictions,
                                          input_decoded]]

            if config.neuron.use_wandb:
                if epoch % 5000 == 0:
                    wandb_table = wandb.Table(columns=['epoch',
                                                       'distil_target_acc',
                                                       'distil_predictions', 'distil2_predictions',
                                                       'teacher_predictions', 'input'])
                    for row in wandb_table_data:
                        wandb_table.add_data(*row)
                    wandb.log({'training_samples': wandb_table})

                    torch.save(distil_model.state_dict(), "{}/distil_model_3.torch".format(config.wandb.directory))
                    torch.save(distil2_model.state_dict(), "{}/distil2_model_3.torch".format(config.wandb.directory))

                wandb.log({'distil_loss_clm': distil_loss_clm.item(),
                           'distil_loss_mse': distil_loss_mse.item(),
                           'distil_loss_ce': distil_loss_ce.item(),
                           'distil_loss_cos': distil_loss_cos.item(),
                           'distil_remote_target_loss': distil_remote_target_loss.item(),
                           'distil_total_loss': distil_total_loss.item(),
                           'distil2_loss_clm': distil2_loss_clm.item(),
                           'distil2_loss_mse': distil2_loss_mse.item(),
                           'distil2_loss_ce': distil2_loss_ce.item(),
                           'distil2_loss_cos': distil2_loss_cos.item(),
                           'distil2_remote_target_loss': distil2_remote_target_loss.item(),
                           'distil2_total_loss': distil2_total_loss.item(),
                           'random_loss': random_loss.item(),
                           'distil_acc': distil_acc,
                           'distil2_acc': distil2_acc,
                           'random_acc': random_acc,
                           'distil_target_acc': distil_target_acc,
                           'distil_remote_target_acc': distil_remote_target_acc,
                           'distil2_target_acc': distil2_target_acc,
                           'distil2_remote_target_acc': distil2_remote_target_acc,
                           'teacher_target_acc': teacher_target_acc,
                           'random_target_acc': random_target_acc,
                           'distil_lr': distil_lr,
                           'distil2_lr': distil2_lr,
                           'random_lr': random_lr})

        torch.cuda.empty_cache()

        distil_total_loss.backward()  # accumulate gradients wrt training loss
        distil2_total_loss.backward()  # accumulate gradients wrt training loss
        random_loss.backward()  # accumulate gradients wrt training loss

        if epoch % config.nucleus.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(distil_model.parameters(), 0.5)
            distil_optimizer.step()  # update model parameters to reduce loss
            distil_optimizer.zero_grad()  # remove previously accumulated gradients
            if distil_scheduler:
                distil_scheduler.step()  # update learning rate multiplier

            torch.nn.utils.clip_grad_norm_(distil2_model.parameters(), 0.5)
            distil2_optimizer.step()  # update model parameters to reduce loss
            distil2_optimizer.zero_grad()  # remove previously accumulated gradients
            if distil2_scheduler:
                distil2_scheduler.step()  # update learning rate multiplier

            torch.nn.utils.clip_grad_norm_(random_model.parameters(), 0.5)
            random_optimizer.step()  # update model parameters to reduce loss
            random_optimizer.zero_grad()  # remove previously accumulated gradients
            if random_scheduler:
                random_scheduler.step()  # update learning rate multiplier

        torch.cuda.empty_cache()


if __name__ == '__main__':
    use_config = main_config()
    main(use_config)
