from __future__ import print_function
from argparse import Namespace
import os
import pickle
import sys
from datetime import datetime
import torch

from src.utils.segmentation.general import set_segmentation_input_channels
from src.utils.cluster.general import config_to_str
from .utility_funcs import setup_config, result_log, load, training, evaluation


"""
  Fully unsupervised clustering for segmentation ("IIC" = "IID").
  Train and test script.
  Network has two heads, for overclustering and final clustering.
"""


def config_setup(config) -> tuple:
    """Finishes the configuration setup.

    Arguments:
        config {Namespace} -- config parsed from the command line

    Returns:
        Final Configuration {Namespace}
    """
    no_cuda = config.nocuda
    config.out_dir = os.path.join(config.out_root, str(config.model_ind))
    config.dataloader_batch_sz = int(
        config.batch_sz / config.num_dataloaders)
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
        config_name, dict_name = "config.pickle", "latest.pytorch"

        reloaded_config_path = os.path.join(config.out_dir, config_name)
        print("Loading restarting config from: %s" % reloaded_config_path)
        with open(reloaded_config_path, "rb") as config_f:
            config = pickle.load(config_f)
        assert (config.model_ind == config.model_ind)
        config.restart = True
        config.nocuda = no_cuda

        # copy over new num_epochs and lr schedule
        config.num_epochs = config.num_epochs
        config.lr_schedule = config.lr_schedule
    else:
        dict_name = False
        print("Given config: %s" % config_to_str(config))

    return config, dict_name


def train(config, dict_name):

    # Data loading
    dataloaders_head_A, dataloaders_head_B, mapping_assignment_dataloader, mapping_test_dataloader, net, optimiser, heads = load(
        config, dict_name)

    # Results and logs
    next_epoch, fig, axarr, loss_fn = result_log(
        config, net, mapping_assignment_dataloader, mapping_test_dataloader)

    # Epoch iteration
    for e_i in range(next_epoch, config.num_epochs):
        print("Starting e_i: %d %s" % (e_i, datetime.now()))
        sys.stdout.flush()

        # Forward + backward pass
        training(config, net, e_i, next_epoch, heads, dataloaders_head_A,
                 dataloaders_head_B, loss_fn, optimiser)

        # Eval stats and model checkpoint
        evaluation(config, net, optimiser, mapping_assignment_dataloader,
                   mapping_test_dataloader, fig, axarr, e_i)

    return net


# Train loop
parser = setup_config()                   # parsing options
config = parser.parse_args()
config, dict_name = config_setup(config)  # defining the configuration

if config.nocuda:
    net = train(config, dict_name)            # training
else:
    torch.cuda.empty_cache()
    with torch.cuda.device(0):
        torch.cuda.empty_cache()
        net = train(config, dict_name)
