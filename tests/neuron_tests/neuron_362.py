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

""" Pretrained server adaptor and validator tests
"""

import wandb
import torch
import argparse
import bittensor

from datasets import load_dataset
from torch import nn
from torch.utils.data import DataLoader
from transformers import GPT2TokenizerFast, GPT2LMHeadModel


def modify_args(parser: argparse.ArgumentParser):
    r""" Modify custom params in the parser for this test.
    """
    parser.add_argument('--wandb.name', type=str, help='''wandb run name''', default='gpt2-xl')
    parser.add_argument('--wandb.run_group', type=str, help='''Optionally pass wandb group name for use_wandb''',
                        default='adaptor-362-0')
    parser.add_argument('--wandb.project', type=str, help='''Optionally pass wandb project name for use_wandb''',
                        default='neuron-tests-adaptor')
    parser.add_argument('--wandb.tags', type=str, help='''Optionally pass wandb tags for use_wandb''',
                        default='pretraining server adaptor, pretrain validator taskhead')

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

    parser.add_argument('--neuron.learning_rate', type=float, help='Training initial learning rate.', default=1e-4)
    parser.add_argument('--neuron.weight_decay', type=float, help='nucleus parameter weight decay.', default=0.25)
    parser.add_argument('--neuron.momentum', type=float, help='optimizer momentum.', default=0.8)
    parser.add_argument('--neuron.clip_gradients', type=float,
                        help='Implement gradient clipping to avoid exploding loss on smaller architectures.',
                        default=1.0)
    parser.add_argument('--neuron.batch_size_train', type=int, help='Training batch size.', default=32)
    parser.add_argument('--neuron.device', type=str, help='Torch device for training.', default='cuda:1')
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


def main(config: 'bittensor.Config'):
    r"""
    Pretrained server adaptor and validator tests.
    Args:
        config (:obj:`bittensor.Config`, `required`): bittensor config

    Returns:

    """
    standard_head = GPT2LMHeadModel.from_pretrained('gpt2-medium').lm_head.state_dict()
    standard_vocab_dim = standard_head['weight'].shape[0]

    model_name = config.wandb.name
    device = config.neuron.device

    # remote = GPT2LMHeadModel.from_pretrained(model_name).half().to(device)
    remote = GPT2LMHeadModel.from_pretrained(model_name).to(device)
    remote_head = nn.Linear(remote.lm_head.in_features, remote.lm_head.out_features, bias=False).to(device)
    remote_head.load_state_dict(remote.lm_head.state_dict())
    remote_head = remote_head.float()
    remote_dim = remote.config.n_embd
    print('remote model:', remote.name_or_path, remote_dim)

    adapt_norm = nn.Linear(remote.config.n_embd, bittensor.__network_dim__, bias=False).to(device)
    adapt_init = nn.Linear(remote.config.n_embd, bittensor.__network_dim__, bias=False).to(device)
    adapt_fine = nn.Linear(remote.config.n_embd, bittensor.__network_dim__, bias=False).to(device)
    adapt_fine2 = nn.Linear(remote.config.n_embd, bittensor.__network_dim__, bias=False).to(device)

    val_norm = nn.Linear(bittensor.__network_dim__, standard_vocab_dim, bias=False).to(device)
    val_init = nn.Linear(bittensor.__network_dim__, standard_vocab_dim, bias=False).to(device)
    val_fine = nn.Linear(bittensor.__network_dim__, standard_vocab_dim, bias=False).to(device)
    val_fine2 = nn.Linear(bittensor.__network_dim__, standard_vocab_dim, bias=False).to(device)

    for linear in [adapt_norm, val_norm]:
        linear.weight.data.normal_(mean=0.0, std=remote.config.initializer_range)

    val_init.load_state_dict(standard_head)
    val_init.weight.requires_grad = False
    val_fine.load_state_dict(standard_head)
    val_fine2.load_state_dict(standard_head)

    opt_params = (list(adapt_norm.parameters()) + list(adapt_fine.parameters()) + list(adapt_fine2.parameters()) +
                  list(val_norm.parameters()) + list(val_fine.parameters()) + list(val_fine2.parameters()))
    optimizer = torch.optim.AdamW(opt_params, lr=config.neuron.learning_rate)

    # xR = yV  : [1, n] x [n, m] = [1, w] x [w, m]
    # xRV' = yVV'
    # xRV' = y : [1, n] -> [1, w]
    # x: [1, n] remote embeddings
    # R: [n, m] remote taskhead
    # y: [1, w] resized remote embeddings
    # V: [w, m] validator taskhead

    V = val_init.weight.data.T  # [1024, 50257]
    Vi = torch.linalg.pinv(V)  # [50257, 1024]

    R = remote.lm_head.weight.data.T.float()

    with torch.no_grad():
        # determine adaptor S = RV': invert V with gradient descent
        S = torch.matmul(R, Vi).T
        print(adapt_init.weight.data.shape, S.shape)

        adapt_init.weight.data = S.clone()
        adapt_fine.weight.data = S.clone()
        adapt_fine2.weight.data = S.clone()

    # torch.save({'S': S}, "{}-adaptor.torch".format(config.wandb.name))

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

    if config.neuron.use_wandb:
        bittensor.wandb(config)  # Initialize wandb logging

    for dataset_epoch in range(1):
        # Define pytorch dataloader with shuffled batches of batch_size token sequences of block_size length.
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch, batch in enumerate(dataloader):
            wandb_log = {'epoch': epoch}

            input_ids = batch['input_ids'].to(device)
            target = input_ids[:, -1]  # held out target of last token
            input_ids = input_ids[:, :-1]  # entire sequence except last token

            with torch.no_grad():
                remote_output = remote(input_ids=input_ids, output_hidden_states=True)
                remote_hidden = remote_output.hidden_states[-1].float().detach()  # x

                # direct model task-head performance
                remote_logits = remote_output.logits.float().contiguous()
                shift_logits = remote_logits[..., :-1, :].contiguous()
                shift_labels = input_ids[..., 1:].contiguous()

                wandb_log['remote_loss_clm'] = nn.CrossEntropyLoss()(shift_logits.view(-1, shift_logits.size(-1)),
                                                                     shift_labels.view(-1)).item()
                predictions = shift_logits.detach().max(2).indices
                wandb_log['remote_acc'] = (predictions == shift_labels).sum().item() / predictions.nelement()

                remote_prediction = remote_logits[:, -1, :].argmax(-1)  # predict unseen last token
                # remote_predictions = tokenizer.decode(remote_output.logits[0].argmax(-1).detach())

                wandb_log['remote_target_acc'] = (remote_prediction == target).sum().item() / len(target)

                # ====================

                # remote hidden -> remote task-head performance
                logits = remote_head(remote_hidden)
                shift_logits = logits[..., :-1, :].contiguous()

                wandb_log['remotehead_loss_clm'] = nn.CrossEntropyLoss()(shift_logits.view(-1, shift_logits.size(-1)),
                                                                     shift_labels.view(-1)).item()
                predictions = shift_logits.detach().max(2).indices
                wandb_log['remotehead_acc'] = (predictions == shift_labels).sum().item() / predictions.nelement()

                remote_prediction = remote_logits[:, -1, :].argmax(-1)  # predict unseen last token
                # remote_predictions = tokenizer.decode(remote_output.logits[0].argmax(-1).detach())

                wandb_log['remotehead_target_acc'] = (remote_prediction == target).sum().item() / len(target)

                # ====================

                # pretrained adapter + validator, but with no further training
                hidden = adapt_init(remote_hidden)  # xRV' = y
                output = val_init(hidden)
                wandb_log['init_logit_mse'] = ((output - remote_logits) ** 2).mean().item()
                shift_logits = output[..., :-1, :].contiguous()
                wandb_log['init_loss_clm'] = nn.CrossEntropyLoss()(shift_logits.view(-1, shift_logits.size(-1)),
                                                                   shift_labels.view(-1)).item()
                predictions = shift_logits.detach().max(2).indices
                wandb_log['init_acc'] = (predictions == shift_labels).sum().item() / predictions.nelement()
                prediction = output[:, -1, :].argmax(-1)  # predict unseen last token
                wandb_log['init_target_acc'] = (prediction == target).sum().item() / len(target)

                # --------

                # xR = yV
                x = remote_hidden
                wandb_log['x.shape'] = str(list(x.shape))
                y = hidden
                wandb_log['y.shape'] = str(list(y.shape))
                xR = torch.matmul(x, R)
                wandb_log['xR.shape'] = str(list(xR.shape))
                yV = torch.matmul(y, V)
                wandb_log['yV.shape'] = str(list(yV.shape))
                wandb_log['MSE(xR,yV)'] = ((xR - yV) ** 2).mean().item()
                I = torch.eye(y.shape[-1]).to(device)
                wandb_log['I.shape'] = str(list(I.shape))
                VVi = torch.matmul(V, Vi)
                wandb_log['VVi.shape'] = str(list(VVi.shape))
                wandb_log['MSE(VVi,I)'] = ((I - VVi) ** 2).mean().item()
                yVVi = torch.matmul(torch.matmul(y, V), Vi)
                wandb_log['yVVi.shape'] = str(list(yVVi.shape))
                wandb_log['MSE(y, (yV)Vi)'] = ((y - yVVi) ** 2).mean().item()
                y_VVi = torch.matmul(y, VVi)
                wandb_log['MSE(y, y(VVi)'] = ((y - y_VVi) ** 2).mean().item()
                yI = torch.matmul(y, I)
                wandb_log['MSE(y, yI)'] = ((y - yI) ** 2).mean().item()

            # normal randomly initialized adapter + validator, with training
            type = 'norm_'
            hidden = adapt_norm(remote_hidden)
            output = val_norm(hidden)
            shift_logits = output[..., :-1, :].contiguous()
            loss = nn.CrossEntropyLoss()(shift_logits.view(-1, shift_logits.size(-1)),
                                                                 shift_labels.view(-1))
            with torch.no_grad():
                wandb_log[type + 'loss_clm'] = loss.item()
                wandb_log[type + 'logit_mse'] = ((output - remote_logits) ** 2).mean().item()
                predictions = shift_logits.detach().max(2).indices
                wandb_log[type + 'acc'] = (predictions == shift_labels).sum().item() / predictions.nelement()
                prediction = output[:, -1, :].argmax(-1)  # predict unseen last token
                wandb_log[type + 'target_acc'] = (prediction == target).sum().item() / len(target)

            torch.cuda.empty_cache()

            # ====================

            # pretrained initialized adapter + validator, with training
            type = 'fine_'
            hidden = adapt_fine(remote_hidden)
            output = val_fine(hidden)
            shift_logits = output[..., :-1, :].contiguous()
            fine_loss = nn.CrossEntropyLoss()(shift_logits.view(-1, shift_logits.size(-1)),
                                              shift_labels.view(-1))
            loss += fine_loss
            with torch.no_grad():
                wandb_log[type + 'loss_clm'] = fine_loss.item()
                wandb_log[type + 'logit_mse'] = ((output - remote_logits) ** 2).mean().item()
                predictions = shift_logits.detach().max(2).indices
                wandb_log[type + 'acc'] = (predictions == shift_labels).sum().item() / predictions.nelement()
                prediction = output[:, -1, :].argmax(-1)  # predict unseen last token
                wandb_log[type + 'target_acc'] = (prediction == target).sum().item() / len(target)

            torch.cuda.empty_cache()

            # ====================

            # pretrained initialized adapter + validator, with training validator
            type = 'val_fine_'
            hidden = adapt_init(remote_hidden)
            output = val_fine2(hidden)
            shift_logits = output[..., :-1, :].contiguous()
            fine_loss = nn.CrossEntropyLoss()(shift_logits.view(-1, shift_logits.size(-1)),
                                              shift_labels.view(-1))
            loss += fine_loss
            with torch.no_grad():
                wandb_log[type + 'loss_clm'] = fine_loss.item()
                wandb_log[type + 'logit_mse'] = ((output - remote_logits) ** 2).mean().item()
                predictions = shift_logits.detach().max(2).indices
                wandb_log[type + 'acc'] = (predictions == shift_labels).sum().item() / predictions.nelement()
                prediction = output[:, -1, :].argmax(-1)  # predict unseen last token
                wandb_log[type + 'target_acc'] = (prediction == target).sum().item() / len(target)

            torch.cuda.empty_cache()

            # ====================

            # pretrained initialized adapter + validator, with training adaptor
            type = 'adapt_fine_'
            hidden = adapt_fine2(remote_hidden)
            output = val_init(hidden)
            shift_logits = output[..., :-1, :].contiguous()
            fine_loss = nn.CrossEntropyLoss()(shift_logits.view(-1, shift_logits.size(-1)),
                                              shift_labels.view(-1))
            loss += fine_loss
            with torch.no_grad():
                wandb_log[type + 'loss_clm'] = fine_loss.item()
                wandb_log[type + 'logit_mse'] = ((output - remote_logits) ** 2).mean().item()
                predictions = shift_logits.detach().max(2).indices
                wandb_log[type + 'acc'] = (predictions == shift_labels).sum().item() / predictions.nelement()
                prediction = output[:, -1, :].argmax(-1)  # predict unseen last token
                wandb_log[type + 'target_acc'] = (prediction == target).sum().item() / len(target)

            torch.cuda.empty_cache()

            with torch.no_grad():
                if epoch % 100 == 0:
                    print(wandb_log)

                wandb.log(wandb_log)

            loss.backward()  # accumulate gradients wrt training loss

            if epoch % config.nucleus.gradient_accumulation_steps == 0:
                optimizer.step()  # update model parameters to reduce loss
                optimizer.zero_grad()  # remove previously accumulated gradients

            torch.cuda.empty_cache()


if __name__ == '__main__':
    use_config = main_config()
    main(use_config)
