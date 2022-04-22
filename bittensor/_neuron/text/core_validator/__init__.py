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
""" The bittensor base validator

Example:
    $ python3 miners/text/core_validator.py --logging.debug

"""
import sys
import argparse
import time
from types import SimpleNamespace
import bittensor
import torch
import os
import wandb
import math
import pandas
import traceback
from rich import print
from rich.console import Console
from rich.traceback import install
from ..neuron_utilities import joining_context, partial_contexts, leave_one_in_partial_contexts, ThreadQueue
import torch.nn as nn
import random
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from loguru import logger
import cProfile
from threading import Lock

logger = logger.opt( colors=True )
console = Console()
install(show_locals=True)

_known_random = [  10,   18,   31,   32,   36,   37,   39,   73,   80,  110,  117,  119,
         121,  124,  130,  135,  145,  149,  193,  208,  212,  225,  253,  255,
         268,  278,  288,  291,  298,  328,  334,  335,  361,  365,  404,  405,
         412,  416,  425,  430,  435,  440,  450,  457,  473,  477,  504,  507,
         517,  532,  533,  535,  549,  562,  579,  591,  600,  604,  633,  636,
         638,  646,  651,  654,  656,  662,  686,  692,  697,  698,  718,  729,
         738,  765,  769,  772,  780,  784,  790,  794,  820,  821,  836,  843,
         857,  859,  868,  876,  884,  911,  916,  928,  930,  937,  944,  945,
         947,  951,  996, 1000, 1002, 1010, 1044, 1071, 1105, 1109, 1111, 1113,
        1123, 1162, 1177, 1180, 1184, 1196, 1200, 1206, 1215, 1222, 1227, 1228,
        1238, 1247, 1256, 1264, 1270, 1304, 1309, 1356, 1371, 1390, 1399, 1405,
        1410, 1414, 1417, 1422, 1432, 1445, 1449, 1455, 1458, 1470, 1486, 1497,
        1498, 1502, 1526, 1541, 1551, 1557, 1563, 1598, 1625, 1630, 1651, 1673,
        1677, 1680, 1683, 1698, 1700, 1709, 1717, 1720, 1804, 1817, 1821, 1828,
        1839, 1851, 1868, 1869, 1870, 1889, 1900, 1922, 1924, 1941, 1959, 1968,
        1985, 2004, 2005, 2011, 2014, 2024, 2026, 2032, 2033, 2036, 2046, 2048,
        2051, 2052, 2054, 2058, 2059, 2061, 2062, 2063, 2065, 2067, 2069, 2071,
        2077, 2082, 2084, 2086, 2087, 2089, 2090, 2091, 2092, 2093, 2094, 2095,
        2102, 2103, 2104, 2106, 2107, 2108, 2109, 2110, 2111, 2113, 2116, 2117,
        2118, 2121, 2122, 2125, 2128, 2130, 2137, 2138, 2139, 2145, 2147, 2148,
        2149, 2158, 2159, 2163, 2164, 2167, 2168, 2169, 2170, 2171, 2172, 2173,
        2174, 2182, 2183, 2184, 2185, 2192, 2193, 2194, 2195, 2196, 2200, 2202,
        2203, 2206, 2210, 2211, 2212, 2213, 2215, 2216, 2217, 2223, 2224, 2230,
        2235, 2239, 2240, 2242, 2243, 2244, 2245, 2247, 2249, 2253, 2254, 2255,
        2260, 2261, 2265, 2266, 2267, 2268, 2270, 2271, 2272, 2275, 2283, 2284,
        2285, 2286, 2287, 2289, 2290, 2293, 2294, 2299, 2300, 2301, 2302, 2304,
        2305, 2306, 2307, 2309, 2310, 2312, 2313, 2316, 2322, 2323, 2324, 2325,
        2328, 2330, 2333, 2338, 2339, 2340, 2341, 2346, 2347, 2348, 2349, 2350,
        2352, 2354, 2355, 2357, 2360, 2362, 2364, 2368, 2369, 2370, 2371, 2373,
        2374, 2375, 2376, 2377, 2378, 2379, 2380, 2382, 2383, 2386, 2387, 2388,
        2389, 2390, 2395, 2397, 2400, 2406, 2407, 2408, 2412, 2413, 2414, 2415,
        2417, 2419, 2420, 2421, 2422, 2423, 2424, 2427, 2429, 2432, 2436, 2437,
        2438, 2439, 2441, 2443, 2444, 2450, 2452, 2454, 2459, 2462, 2468, 2469,
        2471, 2472, 2473, 2475, 2481, 2483, 2486, 2487, 2488, 2489, 2490, 2491,
        2492, 2493, 2495, 2498, 2499, 2500, 2501, 2506, 2507, 2510, 2513, 2515,
        2519, 2520, 2528, 2529, 2536, 2538, 2542, 2543, 2545, 2546, 2548, 2549,
        2550, 2551, 2553, 2555, 2556, 2559, 2560, 2561, 2562, 2563, 2564, 2570,
        2571, 2572, 2573, 2574, 2575, 2576, 2577, 2579, 2580, 2581, 2586, 2589,
        2590, 2591, 2592, 2594, 2595, 2596, 2599, 2605, 2607, 2610, 2611, 2612,
        2613, 2614, 2617, 2618, 2621, 2622, 2624, 2625, 2626, 2627, 2629, 2631,
        2632, 2633, 2638, 2639, 2640, 2642, 2643, 2645, 2646, 2647, 2650, 2651,
        2655, 2656, 2657, 2658, 2659, 2662, 2663, 2664, 2665, 2666, 2669, 2670,
        2671, 2672, 2673, 2674, 2675, 2676, 2678, 2679, 2680, 2682, 2683, 2684,
        2685, 2686, 2687, 2688, 2690, 2693, 2694, 2695, 2698, 2699, 2700, 2701,
        2702, 2706, 2707, 2708, 2711, 2712, 2713, 2715, 2716, 2717, 2718, 2719,
        2720, 2723, 2724, 2725, 2726, 2727, 2729, 2730, 2731, 2732, 2733, 2734,
        2739, 2745, 2747, 2748, 2750, 2751, 2752, 2753, 2754, 2756, 2757, 2758,
        2760, 2761, 2762, 2764, 2765, 2767, 2768, 2771, 2772, 2773, 2774, 2776,
        2777, 2778, 2779, 2780, 2782, 2783, 2784, 2785, 2787, 2788, 2789, 2790,
        2791, 2792, 2794, 2795, 2796, 2797, 2799, 2803, 2804, 2805, 2807, 2810,
        2817, 2818, 2819, 2820, 2822, 2823, 2824, 2825, 2826, 2828, 2829, 2831,
        2833, 2834, 2835, 2836, 2838, 2839, 2842, 2843, 2847, 2848, 2851, 2854,
        2856, 2860, 2864, 2865, 2867, 2868, 2870, 2871, 2875, 2876, 2877, 2878,
        2879, 2880, 2882, 2883, 2884, 2885, 2888, 2889, 2890, 2891, 2892, 2893,
        2895, 2898, 2899, 2900, 2901, 2902, 2903, 2904, 2905, 2906, 2907, 2908,
        2910, 2911, 2915, 2916, 2917, 2918, 2920, 2922, 2923, 2924, 2925, 2926,
        2927, 2930, 2935, 2936, 2937, 2938, 2939, 2940, 2941, 2942, 2943, 2944,
        2945, 2946, 2947, 2948, 2949, 2950, 2951, 2955, 2956, 2958, 2959, 2960,
        2961, 2962, 2965, 2966, 2967, 2969, 2970, 2971, 2972, 2973, 2976, 2977,
        2978, 2980, 2981, 2982, 2983, 2987, 2988, 2989, 2991, 2993, 2995, 2996,
        2999, 3001, 3002, 3003, 3004, 3009, 3010, 3011, 3012, 3013, 3015, 3016,
        3017, 3021, 3022, 3024, 3025, 3026, 3027, 3029, 3032, 3033, 3034, 3035,
        3036, 3037, 3039, 3043, 3045, 3046, 3048, 3050, 3051, 3052, 3058, 3059,
        3060, 3061, 3066, 3067, 3069, 3071, 3072, 3075, 3076, 3077, 3078, 3079,
        3081, 3082, 3083, 3084, 3085, 3086, 3087, 3088, 3089, 3090, 3091, 3092,
        3093, 3094, 3099, 3100, 3101, 3102, 3103, 3106, 3108, 3109, 3110, 3111,
        3112, 3113, 3114, 3115, 3116, 3117, 3118, 3119, 3120, 3121, 3122, 3123,
        3124, 3125, 3128, 3129, 3130, 3131, 3133, 3134, 3136, 3137, 3138, 3139,
        3140, 3144, 3145, 3146, 3147, 3148]

class neuron:
    r"""
    Creates a bittensor neuron that specializes validating other peers. The core validator
    finetunes on the bittensor network with a mixture of experts model and shapely scoring.
    The validator's main jobs are to identify important/useful peers in the network and correctly
    weight them. To achieve this, the validator will send requests to different peers on the network
    and evalute their responses.

    Args: 
            config (:obj:`bittensor.Config`, `optional`): 
                bittensor.server.config()
            subtensor (:obj:bittensor.subtensor , `optional`):
                bittensor subtensor connection
            dataset (:obj:bittensor.dataset , `optional`):
                bittensor dataset 
            wallet (:obj:bittensor.wallet, `optional`):
                bittensor wallet object
            metagraph (:obj:bittensor.metagraph, `optional`):
                bittensor metagraph object
            dendrite (:obj:bittensor.dendrite, `optional`):
                bittensor dendrite object
            dataset (:obj:bittensor.dendrite, `optional`):
                bittensor dendrite object
    Examples:: 
            >>> subtensor = bittensor.subtensor(network='nakamoto')
            >>> validator = bittensor.neuron.text.core_validator.neuron(subtensor=subtensor)
            >>> validator.run()
    """
    def __init__( 
        self, 
        config: 'bittensor.Config' = None,
        wallet: 'bittensor.Wallet' = None,
        subtensor: 'bittensor.Subtensor' = None,
        metagraph: 'bittensor.Metagraph' = None,
        dendrite: 'bittensor.Dendrite' = None,
        dataset: 'bittensor.dataset' = None
    ):

        # === Set up Config ===
        if config == None: config = neuron.config()
        self.config = config
        neuron.check_config( self.config )
        self.config.to_defaults()
        if self.config.neuron._mock == True:
            self.config.subtensor._mock = True
            self.config.wallet._mock = True
            self.config.dataset._mock = True
            self.config.dendrite._mock = True
            self.config.metagraph._mock = True
            self.config.subtensor._mock = True
        print ( self.config )

        # === Create Bittensor objects ===
        bittensor.logging( config = self.config, logging_dir = self.config.neuron.full_path )
        self.wallet = bittensor.wallet ( config = self.config ) if wallet == None else wallet
        self.subtensor = bittensor.subtensor ( config = self.config ) if subtensor == None else subtensor
        self.metagraph = bittensor.metagraph ( config = config, subtensor = self.subtensor ) if metagraph == None else metagraph
        self.dendrite = bittensor.dendrite ( config = self.config, wallet = self.wallet ) if dendrite == None else dendrite
        self.device = torch.device ( device = self.config.neuron.device )    
        self.nucleus = nucleus ( config = self.config, device = self.device, subtensor = self.subtensor ).to( self.device )
        self.dataset = bittensor.dataset ( config = self.config, batch_size = self.subtensor.validator_batch_size, block_size = self.subtensor.validator_sequence_length ) if dataset == None else dataset
        
        # === Create thread queue ===
        self.forward_thread_queue = ThreadQueue(num_jobs = self.config.neuron.forward_num, target = self.forward)
        self.loss = None
        self.loss_agg_mutex = Lock()
        self.moving_avg_scores = None

    @classmethod
    def check_config( cls, config: 'bittensor.Config' ):
        r""" Checks/validates the config namespace object.
        """
        nucleus.check_config( config )
        bittensor.logging.check_config( config )
        bittensor.wallet.check_config( config )
        bittensor.subtensor.check_config( config )
        bittensor.metagraph.check_config( config )
        bittensor.dataset.check_config( config )
        bittensor.dendrite.check_config( config )
        bittensor.wandb.check_config( config )
        full_path = os.path.expanduser('{}/{}/{}/{}'.format( config.logging.logging_dir, config.wallet.name, config.wallet.hotkey, config.neuron.name ))
        config.neuron.full_path = os.path.expanduser(full_path)
        config.using_wandb = config.wandb.api_key != 'default'
        if not os.path.exists(config.neuron.full_path):
            os.makedirs(config.neuron.full_path)

    @classmethod
    def add_args( cls, parser ):
        parser.add_argument('--neuron.name', type=str, help='Trials for this miner go in miner.root / (wallet_cold - wallet_hot) / miner.name ', default='core_validator')
        parser.add_argument('--neuron.learning_rate', type=float, help='Training initial learning rate.', default=0.1 )
        parser.add_argument('--neuron.momentum', type=float, help='optimizer momentum.', default=0.8 )
        parser.add_argument('--neuron.blocks_per_epoch', type=int, help='Blocks per epoch, -1 value means we use the chain value.', default = -1 )
        parser.add_argument('--neuron.epochs_until_reset', type=int, help='Number of epochs before weights are reset.', default = -1 )
        parser.add_argument('--neuron.device', type=str, help='miner default training device cpu/cuda', default=("cuda" if torch.cuda.is_available() else "cpu"))
        parser.add_argument('--neuron.clip_gradients', type=float, help='Implement gradient clipping to avoid exploding loss on smaller architectures.', default=1.0 )
        parser.add_argument('--neuron.restart_on_failure',  action='store_true', help='''Restart neuron on unknown error.''', default=True )
        parser.add_argument('--neuron._mock', action='store_true', help='To turn on neuron mocking for testing purposes.', default=False )
        parser.add_argument('--neuron.wait_for_finalization', action='store_true', help='''when setting weights the miner waits for trnasaction finalization.''', default=False)
        parser.add_argument('--neuron.forward_num', type=int, help='''How much forward request before a backward call.''', default=3)

    @classmethod
    def config ( cls ):
        parser = argparse.ArgumentParser()    
        cls.add_args( parser )
        nucleus.add_args( parser )        
        bittensor.wallet.add_args( parser )
        bittensor.dendrite.add_args( parser )
        bittensor.subtensor.add_args( parser )
        bittensor.metagraph.add_args( parser )
        bittensor.logging.add_args( parser )
        bittensor.dataset.add_args( parser )
        bittensor.wandb.add_args(parser)
        return bittensor.config( parser )
    
    def __del__(self):
        self.__exit__()

    def __exit__ ( self, exc_type, exc_value, exc_traceback ):
        r""" Close down neuron.
        """
        print(exc_type, exc_value, exc_traceback)
        self.dataset.close()
        self.dendrite.__del__()
        self.forward_thread_queue.stop()
        self.forward_thread_queue.join()

    def __enter__(self):
        r""" Sanity checks and begin validator.
        """
        # === Wallet ===
        # Connects wallett to network. 
        # NOTE: This registration step should likely be solved offline first.
        self.wallet.create().register( subtensor = self.subtensor )

        # === UID ===
        # Get our uid from the chain. 
        # At this point we should have a uid because we are already registered.
        self.uid = self.wallet.get_uid( subtensor = self.subtensor )    

        # === Monitoring ===
        # Optionally set up wandb logging.
        if self.config.using_wandb:
            bittensor.wandb(
                config = self.config,
                cold_pubkey = self.wallet.coldkeypub.ss58_address,
                hot_pubkey = self.wallet.hotkey.ss58_address,
                root_dir = self.config.neuron.full_path
            )

    def forward(self):
        r""" Run the nucleus forward request
        This function is supposed to be ran multi-threaded.
        """
        total_loss, scores, routing_uids = self.nucleus( next(self.dataset) , self.metagraph, self.dendrite )
                
        # === Backward ===
        # Backwards gradients through model to train gating and remote endpoints.
        (total_loss / self.config.neuron.forward_num).backward()
        return total_loss, scores, routing_uids

    def run ( self ):
        r""" Run the validator and terminate on Keyboard interrupt.
        """
        # === Setup ===
        # Checks wallet and starts monitoring with wandb.
        with self:

            # === Start forward requests ===
            self.metagraph_sync()
            self.forward_thread_queue.start()
            
            # === Run ===
            # Iterates through epochs.
            self.epoch = 0
            self.global_step = 0
            while True:
                try:

                    # === Epoch ===
                    # Each epoch runs for blocks_per_epoch and resets
                    # the model every epochs_until_reset.
                    self.run_epoch()

                # === Stops on interrupt otherwise restarts ===
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    console.print_exception(show_locals=False)
                    print( traceback.format_exc() )
                    print( 'Unknown exception: {}', e )
                    if not self.config.neuron.restart_on_failure:
                        break

    def run_epoch( self ):
        r""" Runs a validator epoch. We apply batches until the epoch length is exhausted.
            Occasionally the validator nucleus is completely reset to ensure we dont converge to far.
            At the end of the epoch we set weights on the chain and optionally log to wandb.
        """
        # === Get params for epoch ===
        # Pulling the latest chain parameters.
        current_block = self.subtensor.block
        batch_size = self.subtensor.validator_batch_size 
        sequence_length = self.subtensor.validator_sequence_length
        n_topk_peer_weights = self.subtensor.min_allowed_weights
        max_allowed_ratio = self.subtensor.max_allowed_min_max_ratio
        blocks_per_epoch = self.subtensor.validator_epoch_length if self.config.neuron.blocks_per_epoch == -1 else self.config.neuron.blocks_per_epoch
        epochs_until_reset = self.subtensor.validator_epochs_per_reset if self.config.neuron.epochs_until_reset == -1 else self.config.neuron.epochs_until_reset
        # === Logs ===
        print ( '\nEra:', '\n\t batch_size:', batch_size, '\n\t sequence_length:', sequence_length, '\n\t n_topk_peer_weights:', n_topk_peer_weights,
                '\n\t max_allowed_ratio:', max_allowed_ratio, '\n\t blocks_per_epoch:', blocks_per_epoch, '\n\t epochs_until_reset:', epochs_until_reset, 
                '\n\t until_reset:', self.epoch % epochs_until_reset, '\n\t current_block:', current_block, '\n')
        if self.config.using_wandb:
            wandb.log( {    'era/batch_size': batch_size, 'era/sequence_length': sequence_length, 'era/n_topk_peer_weights': n_topk_peer_weights, 
                            'era/max_allowed_ratio': max_allowed_ratio, 'era/blocks_per_epoch': blocks_per_epoch, 'era/epochs_until_reset': epochs_until_reset, 
                }, step = current_block )

        # === Run Epoch ===
        # Each block length lasts blocks_per_epoch blocks.
        # This gives us a consistent network wide timer.
        # Here we run until blocks_per_epochs have progressed.
        self.metagraph_sync() # Reset metagraph.
        epoch_steps = 0

        # === Reset Epochs with new params. ===
        # Pulls new default validator training parameters and resets 
        # the model and dataset for the following epoch.
        if self.epoch % epochs_until_reset == 0:
            print ('\n\n=== Reset ===\n\n')
            # === Resetting model + dataset ===
            if (batch_size != self.dataset.batch_size) or (sequence_length != self.dataset.block_size):
                self.dataset.set_data_size(batch_size, sequence_length)

            self.nucleus = nucleus ( config = self.config, device = self.device, subtensor = self.subtensor ).to( self.device )
            self.optimizer = torch.optim.SGD ( 
                self.nucleus.parameters(), lr = self.config.neuron.learning_rate, momentum = self.config.neuron.momentum 
            )

            # === Reset Scores ===
            self.moving_avg_scores = torch.ones_like( self.metagraph.S ) * -1

        # Checks if moving avg has been initiated
        if self.moving_avg_scores == None:
            self.moving_avg_scores = torch.ones_like( self.metagraph.S ) * -1

        start_block = self.subtensor.block
        while self.subtensor.block < start_block + blocks_per_epoch:
            start_time = time.time()

            # === Forward ===
            # Forwards inputs through the network and returns the loss
            # and endpoint scores using shapely approximation of salience.
            loss, scores, uids = self.forward_thread_queue.get()
            print(f'Run\t| Got forward result in {round(time.time() - start_time, 3)}')

            # === Scoring ===
            # Updates moving averages and history.
            self.moving_avg_scores[uids] = self.moving_avg_scores[uids]*(0.99) + scores*(0.01)
            with open("scores.txt", "a") as scores:
                scores.write( "{} - ".format(self.subtensor.block) + str(list( zip( self.moving_avg_scores.tolist(), self.metagraph.uids.tolist() ) )) + "\n")

        
            # === State update ===
            # Prints step logs to screen.
            epoch_steps += 1
            self.global_step += 1
            current_block = self.subtensor.block
            step_time = time.time() - start_time

            # === Logs ===
            print( '\nStep:', '\n\t epoch:', self.epoch, '\n\t epoch_steps:', epoch_steps, '\n\t global_steps:', self.global_step, '\n\t step_time:', step_time, '\n\t loss:', loss.item(),
                   '\n\t current_block', current_block, '\n\t blocks remaining:', current_block - start_block, '/', blocks_per_epoch, '\n')
            if self.config.using_wandb:
                wandb.log( { 'epoch/epoch': self.epoch, 'epoch/epoch_steps': epoch_steps, 'epoch/global_steps': self.global_step, 'epoch/loss': loss.item(), 'epoch/time': step_time }, step = current_block )
                step_topk_scores, step_topk_uids = bittensor.unbiased_topk( self.moving_avg_scores, k = n_topk_peer_weights )
                step_topk_normalized = bittensor.utils.weight_utils.normalize_max_multiple( x = step_topk_scores, multiple = max_allowed_ratio )
                for i, w in list(zip(step_topk_uids.tolist(), step_topk_normalized.tolist()) ):
                    wandb.log( {'weights/w_{}'.format( i ): w }, step = current_block )

            # Do the backward request after the a queue of forward requests got finished.  
            if self.forward_thread_queue.paused() and self.forward_thread_queue.is_empty():
                print('Run\t| Model update')

                # === Apply gradients ===
                # Applies local gradients to parameters.
                clip_grad_norm_(self.nucleus.parameters(), self.config.neuron.clip_gradients)
                self.optimizer.step()
                self.optimizer.zero_grad()    
                
                # === Get another round of forward requests ===
                self.forward_thread_queue.resume()

        # Iterate epochs.
        self.epoch += 1

        # === Set weights ===
        # Find the n_topk_peer_weights peers to set weights to.
        # We use the mean of the epoch weights.
        topk_scores, topk_uids = bittensor.unbiased_topk(self.moving_avg_scores, k = n_topk_peer_weights )
        topk_scores = bittensor.utils.weight_utils.normalize_max_multiple( x = topk_scores, multiple = max_allowed_ratio )
        print( '\nScores:', '\n\t weights:', topk_scores.sort()[0].tolist(), '\n\t sum:', topk_scores.sum().item(), 
                '\n\t min:', topk_scores.min().item(), '\n\t max:', topk_scores.max().item(), '\n\t max/min:', (topk_scores.max()/topk_scores.min()).item() )
        self.subtensor.set_weights(
            uids = topk_uids.detach().to('cpu'),
            weights = topk_scores.detach().to('cpu'),
            wallet = self.wallet,
            wait_for_finalization = self.config.neuron.wait_for_finalization,
        )

        # === Wandb Logs ===
        # Optionally send validator logs to wandb.
        if self.config.using_wandb:
            # Logging history to wandb.
            df = pandas.concat( [
                bittensor.utils.indexed_values_to_dataframe( prefix = 'weights', index = topk_uids, values = torch.zeros( self.metagraph.n ).scatter( dim = 0, src = topk_scores, index = topk_uids ) ),
                self.dendrite.to_dataframe( metagraph = self.metagraph )
            ], axis = 1); df['uid'] = df.index
            wandb_data_dend = self.dendrite.to_wandb()
            wandb_data = { 'stake': self.metagraph.S[ self.uid ].item(), 'dividends': self.metagraph.D[ self.uid ].item() } 
            wandb.log( { 'stats': wandb.Table( dataframe = df ) }, step = current_block )
            wandb.log( { **wandb_data, **wandb_data_dend }, step = current_block )
    
    def metagraph_sync(self):
        r""" Syncing metagraph together with other metagraph-size related objects
        """
        self.metagraph.sync()
        
        if self.moving_avg_scores == None:
            self.moving_avg_scores = torch.ones_like( self.metagraph.S ) * -1
        
        if self.metagraph.n > len(self.moving_avg_scores):
            size_incerease = self.metagraph.n - len(self.moving_avg_scores)
            self.moving_avg_scores = torch.concat([self.moving_avg_scores, torch.ones(size_incerease) * -1]) 

class PositionalEncoding(nn.Module):
    r""" Positional Encoder which adds information based on the relative position of each token
    
    """
    def __init__(self, d_model: int, dropout: float, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        # === Create position matrix ===
        # Creates a positional matrix with alternating frequencies 
        # pe: (torch.FloatTensor) positional encoding matrix
        # pe.shape: [1, max_len, network_dim]
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, : , 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        # === Positional Encoding ===
        # Inject some information of the relative position of the token in the sequence.
        #  Finally, Dropout is applied to tokens
        # x: (torch.FloatTensor) input sequence tokens with position information injected
        # x.shape: [batch_size, seq_len, network_dim]
        x = x + self.pe[0, :x.size(1)]
        return self.dropout(x)

class nucleus( torch.nn.Module ):
    """ Nucleus class which holds the validator model.
    """
    def __init__( self, config, device, subtensor ):
        super(nucleus, self).__init__()
        self.config = config
        self.device = device
        self.max_n = subtensor.max_n 

        # Token embeddings project int64 tokens onto representations.
        self.token_embedding = torch.nn.Embedding( bittensor.__vocab_size__,  bittensor.__network_dim__ )
        
        # Routing encoder, projects token embeddings onto context for routing inputs.
        self.routing_encoder_layers = TransformerEncoderLayer( bittensor.__network_dim__, config.nucleus.nhead, config.nucleus.nhid, config.nucleus.dropout, batch_first=True)
        self.routing_encoder = TransformerEncoder( self.routing_encoder_layers, 1 )

        # Encoder projects response representations onto hidden units.
        self.encoder_layers = TransformerEncoderLayer( bittensor.__network_dim__, config.nucleus.nhead, config.nucleus.nhid, config.nucleus.dropout, batch_first=True)
        self.encoder = TransformerEncoder( self.encoder_layers, config.nucleus.nlayers )

        # Decoder which projects hidden unit representations on to the token dimension.
        self.decoder = torch.nn.Linear( bittensor.__network_dim__, bittensor.__vocab_size__ , bias=False)

        # Positional Encoding
        self.local_pos_encoder = PositionalEncoding( bittensor.__network_dim__, self.config.nucleus.dropout )

        # Crosss entropy loss for NTP.    
        self.loss_fct = torch.nn.CrossEntropyLoss()
    
        # SGMOE Gates: Instantiating the gates per expert.
        self.gates = torch.nn.Linear( bittensor.__network_dim__, self.max_n, bias=True ).to( self.device )
        self.reset_weights()

    @classmethod
    def add_args( cls, parser ):
        parser.add_argument('--nucleus.topk', type=int, help='the number of peers queried during each remote forward call', default = 20 )
        parser.add_argument('--nucleus.nhid', type=int, help='the dimension of the feedforward network model in nn.TransformerEncoder', default=200 )
        parser.add_argument('--nucleus.nhead', type=int, help='the number of heads in the multiheadattention models', default = 2 )
        parser.add_argument('--nucleus.nlayers', type=int, help='the number of nn.TransformerEncoderLayer in nn.TransformerEncoder', default=2 )
        parser.add_argument('--nucleus.dropout', type=float, help='the dropout value', default=0.2)
        parser.add_argument('--nucleus.importance', type=float, help='hyperparameter for the importance loss', default=3)
        parser.add_argument('--nucleus.noise_multiplier', type=float, help='Standard deviation multipler on weights', default=2 )

    @classmethod
    def config ( cls ):
        parser = argparse.ArgumentParser()    
        cls.add_args( parser )
        return bittensor.config( parser )

    @classmethod
    def check_config( cls, config: 'bittensor.Config' ):
        pass

    def reset_weights ( self ):
        r""" Resets the validator weights.
        """
        # === Resets all the weights using xavier initialization. ===
        torch.nn.init.xavier_uniform_ ( self.token_embedding.weight )
        torch.nn.init.xavier_uniform_ ( self.decoder.weight )
        torch.nn.init.xavier_uniform_( self.gates.weight )
        def init_xavier( component ):
            try:
                torch.nn.init.xavier_uniform_( component.weight )
            except: pass
        self.routing_encoder.apply( init_xavier )
        self.encoder.apply( init_xavier )
        torch.nn.init.xavier_uniform_( self.gates.weight )

    # === Compute loss given joined responses ===
    # This function computes target loss for next token prediction given 
    # the joined responses as a hidden unit input.
    # target_loss: (torch.float64): loss after decoding responses to targets.
    # target_loss.shape = [ 1 ]
    def get_target_loss ( self, hidden, targets ):
        # hidden: (torch.float64): [ batch_size, sequence_len, __network_dim__ ]
        #   Hidden units which are encoded and decoded onto targets for loss computation.
        # targets: (torch.float64): [n]
        #   Token targets,
        src_mask = torch.triu(torch.ones(hidden.size(1), hidden.size(1)) * float('-inf'), diagonal=1)
        src_mask = src_mask.to(self.config.neuron.device)
        encoded_hidden = self.encoder( hidden, mask = src_mask )
        decoded_targets = self.decoder( encoded_hidden )
        shift_logits = decoded_targets[..., :-1, :].contiguous()
        shift_labels = targets[..., 1:].contiguous()
        return self.loss_fct( shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1) )

    def forward ( 
        self, 
        inputs: torch.FloatTensor,
        metagraph: 'bittensor.Metagraph',
        dendrite: 'bittensor.Dendrite',
    ):
        r""" Forward validator pass. Selects peer to query, joins results and computes scoring.
            Args:
                inputs (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, *-1*)`, `required`): 
                    Tensor inputs to distribute to neurons using query context.
                metagraph (bittensor.Metagraph):
                    Metagraph object used to query network information.
                dendrite (bittensor.Dendrite):
                    Dendrite RPC client used to make network queries.
            Returns:
                global_loss (torch.FloatTensor, [1] ):
                    Loss for training validator nucleus.
                scores (torch.FloatTensor, [ metagraph.n ]):
                    Scores per endpoint for this batch.
        """        
        # === Create the local context used to select endpoints ===
        # The context tensor returns a hidden unit representation for the text inputs
        # this context can be used as input to the gates in the next step.
        # embedding: retrieve learned representation vectors for input vocabulary tokens.
        # inputs.shape = [batch_size, sequence_len]
        # embedding.shape = [batch_size, sequence_len, bittensor.__network_dim__]
        embedding =  self.token_embedding( inputs )* math.sqrt( bittensor.__network_dim__ )
        
        # === Create an attention mask ===
        # The attention mask will mask out parts of the context
        # This prevents cheating and forward-looking when predicting each token in the sequence.
        # src_mask: (torch.FloatTensor) attention mask adds -inf to positions not allowed to attend
        # src_mask.shape = [sequence_len, sequence_len]
        src_mask = torch.triu(torch.ones(embedding.size(1), embedding.size(1)) * float('-inf'), diagonal=1)
        src_mask = src_mask.to(self.config.neuron.device)

        # === Apply the positional encoding to help select endpoints ===
        # The positional encoder provides information based on the relative postion of each token 
        # embedding.shape = [batch_size, sequence_len, bittensor.__network_dim__]
        # pos_embedding: (torch.FloatTensor) positional encoded embedding.
        # pos_embedding.shape = [batch_size, sequence_len, bittensor.__network_dim__]
        pos_embedding = self.local_pos_encoder(embedding)

        # routing_context: (torch.FloatTensor): context tensor which is used to select endpoints.
        # routing_context.shape = [ batch size, __network_dim__ ]
        routing_context = self.routing_encoder( pos_embedding, mask = src_mask )

        # === Get weights for uids. ===
        # We iterate over each of the network uids and compute a querying score for each
        # using the gating function. This returns a score per endpoint per example.
        # routing_weights: (torch.FloatTensor): score per example, per endpoint.
        # routing_weights.shape = [ batch size, __network_n__ ]
        # The gates act over the last embedding of the routing_context.
        routing_weights = self.gates( routing_context[:,-1,:] )

        # === Normalize routing_weights across batch dimension and add noise. ===
        # We are summing across the batch dimension to create a per-batch score per endpoint.
        # The resulting routing_weights tensor is a score per expert.
        # routing_weights: (torch.FloatTensor): normalized weights across batch dimension with noise.
        # routing_weights.shape = [ n_filtered ]
        batchwise_routing_weights = torch.mean(routing_weights, axis = 0)[:metagraph.n]
        noisy_routing_weights = torch.normal( 0, torch.std(batchwise_routing_weights).item(), size=( batchwise_routing_weights.size())).to( self.config.neuron.device )
        noisy_routing_weights =  batchwise_routing_weights + noisy_routing_weights * self.config.nucleus.noise_multiplier
        

        # === Get indices and values for uids with highest scores ===
        # We are taking the topk routing weights and returning their uids.
        # First we ensure topk is smaller than the network size then use the torch.topk.
        # topk_routing_weights: (torch.float64): scores of uids with highest scores.
        # topk_routing_weights.shape = [ self.config.nucleus.topk ]
        # topk_routing_uids: (torch.LongTensor): uids with highest scores.
        # topk_routing_uids.shape = [ self.config.nucleus.topk ]
        top_k_routing_weights, routing_uids = torch.topk( noisy_routing_weights, self.config.nucleus.topk, dim=0)

        # === Get endpoint information for the highest scoring uids ===
        # We index into the metagraph's endpoints and return a list of the filtered set of endpoints we wish to query.
        # routing_endpoints: List[bittensor.endpoints]: endpoint information for filtered uids.
        # len(neurons) == self.config.nucleus.topk
        routing_endpoints = [ metagraph.endpoints[ uid ] for uid in routing_uids ]

        # === Query the endpoints ===
        # Makes the dendrite call into the network returning the representations 
        # for each of the endpoints. The return ops can be used to filter weights and outputs.
        # query_responses: (List[torch.float64]): responses from each endpoint.
        # query_responses.shape = self.config.nucleus.topk * [ batch_size, sequence_len, __network_dim__ ]
        # return_ops: (torch.int64): Return ops.
        # return_ops.shape = [ self.config.nucleus.topk ]
        query_responses, return_ops, times = dendrite.forward_text ( 
            endpoints = routing_endpoints, 
            inputs = inputs
        )
        # Send responses to device. This is required to ensure we move the responses
        # Onto the correct device.
        for response in query_responses:
            response.to( self.device )


        scores = torch.zeros( routing_uids.size())
        for i, response in enumerate(query_responses):
            norm_response = response / (response.sum() + 0.000001)
            decoded_targets = self.decoder( response * batchwise_routing_weights[ routing_uids[i] ] )
            shift_logits = decoded_targets[..., :-1, :].contiguous()
            shift_labels = inputs[..., 1:].contiguous()
            loss_i = self.loss_fct( shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1) )
            scores[ i ] = loss_i
            print ('Loss:\t{}  \tuid:\t{}   wieght:{}  rand: {} mean: {} std: {}'.format( loss_i.item(), routing_uids[i], batchwise_routing_weights[ routing_uids[i] ], routing_uids[i] in _known_random, response.mean(), response.std()))

        total_loss = scores.sum()

        # === Done ===
        return total_loss, scores, routing_uids
