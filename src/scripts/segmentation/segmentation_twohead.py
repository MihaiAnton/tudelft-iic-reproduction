from __future__ import print_function
from src.utils.segmentation.general import set_segmentation_input_channels
# from src.utils.segmentation.data import segmentation_create_dataloaders
# from src.utils.segmentation.IID_losses import IID_segmentation_loss, \
#     IID_segmentation_loss_uncollapsed
# from src.utils.segmentation.segmentation_eval import \
#     segmentation_eval
# from src.utils.cluster.transforms import sobel_process
# from src.utils.cluster.general import config_to_str, get_opt, update_lr, nice
from .utility_funcs import *
# import src.archs as archs
import os
# import matplotlib.pyplot as plt

# import argparse
import pickle
import sys
from datetime import datetime

import matplotlib
# import numpy as np
# import torch

# matplotlib.use('Agg')


"""
  Fully unsupervised clustering for segmentation ("IIC" = "IID").
  Train and test script.
  Network has two heads, for overclustering and final clustering.
"""

# Options ----------------------------------------------------------------------

config = parser.parse_args()

# Setup ------------------------------------------------------------------------

no_cuda = config.nocuda
config.out_dir = os.path.join(config.out_root, str(config.model_ind))
config.dataloader_batch_sz = int(config.batch_sz / config.num_dataloaders)
assert (config.mode == "IID")
assert ("TwoHead" in config.arch)
assert (config.output_k_B == config.gt_k)
config.output_k = config.output_k_B  # for eval code
assert (config.output_k_A >= config.gt_k)  # sanity
config.use_doersch_datasets = False
config.eval_mode = "hung"
set_segmentation_input_channels(config)
print(config.nocuda)
if not os.path.exists(config.out_dir):
    os.makedirs(config.out_dir)

if config.restart:
    config_name = "config.pickle"
    dict_name = "latest.pytorch"

    given_config = config
    reloaded_config_path = os.path.join(given_config.out_dir, config_name)
    print("Loading restarting config from: %s" % reloaded_config_path)
    with open(reloaded_config_path, "rb") as config_f:
        config = pickle.load(config_f)
    assert (config.model_ind == given_config.model_ind)
    config.restart = True
    config.nocuda = no_cuda

    # copy over new num_epochs and lr schedule
    config.num_epochs = given_config.num_epochs
    config.lr_schedule = given_config.lr_schedule
else:
    dict_name = False
    print("Given config: %s" % config_to_str(config))


# Model ------------------------------------------------------

def train():

    dataloaders_head_A, dataloaders_head_B, mapping_assignment_dataloader, mapping_test_dataloader, net, optimiser, heads = load(
        config, dict_name)

    # Results
    # ----------------------------------------------------------------------
    next_epoch, fig, axarr, loss_fn = result_log(
        config, net, mapping_assignment_dataloader, mapping_test_dataloader)

    # Train
    # ------------------------------------------------------------------------
    for e_i in range(next_epoch, config.num_epochs):
        print("Starting e_i: %d %s" % (e_i, datetime.now()))
        sys.stdout.flush()

        training(config, net, e_i, next_epoch, heads, dataloaders_head_A,
                 dataloaders_head_B, loss_fn, optimiser)  # forward + backward pass
        # logging eval_stats and model checkpt.
        evaluation(config, net, optimiser, mapping_assignment_dataloader,
                   mapping_test_dataloader, fig, axarr, e_i)


train()
