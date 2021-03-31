from __future__ import print_function
from src.utils.segmentation.general import set_segmentation_input_channels
from src.utils.segmentation.data import segmentation_create_dataloaders
from src.utils.segmentation.IID_losses import IID_segmentation_loss, \
    IID_segmentation_loss_uncollapsed
from src.utils.segmentation.segmentation_eval import \
    segmentation_eval
from src.utils.cluster.transforms import sobel_process
from src.utils.cluster.general import config_to_str, get_opt, update_lr, nice
import src.archs as archs
import matplotlib.pyplot as plt

import argparse
import os
import pickle
import sys
from datetime import datetime

import matplotlib
import numpy as np
import torch

matplotlib.use('Agg')


"""
  Semisupervised overclustering for segmentation ("IIC+" = "IID+")
  Note network is trained entirely unsupervised, as labels are found for 
  evaluation only and do not affect the network.
  Train and test script.
  Network has one output head only.
"""

# Options ----------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--model_ind", type=int, required=True)
parser.add_argument("--arch", type=str, required=True)
parser.add_argument("--opt", type=str, default="Adam")
parser.add_argument("--mode", type=str, default="IID+")

parser.add_argument("--dataset", type=str, required=True)
parser.add_argument("--dataset_root", type=str, required=True)

parser.add_argument("--use_coarse_labels", default=False,
                    action="store_true")  # COCO, Potsdam
parser.add_argument("--fine_to_coarse_dict", type=str,  # COCO
                    default="/users/xuji/iid/iid_private/src/datasets"
                            "/segmentation/util/out/fine_to_coarse_dict.pickle")
parser.add_argument("--include_things_labels", default=False,
                    action="store_true")  # COCO
parser.add_argument("--incl_animal_things", default=False,
                    action="store_true")  # COCO
parser.add_argument("--coco_164k_curated_version",
                    type=int, default=-1)  # COCO

parser.add_argument("--gt_k", type=int, required=True)
parser.add_argument("--output_k", type=int, required=True)

parser.add_argument("--lamb", type=float, default=1.0)
parser.add_argument("--lr", type=float, default=0.01)
parser.add_argument("--lr_schedule", type=int, nargs="+", default=[])
parser.add_argument("--lr_mult", type=float, default=0.1)

parser.add_argument("--use_uncollapsed_loss", default=False,
                    action="store_true")
parser.add_argument("--mask_input", default=False, action="store_true")

parser.add_argument("--num_epochs", type=int, default=1000)
parser.add_argument("--batch_sz", type=int, required=True)  # num pairs
parser.add_argument("--num_dataloaders", type=int, default=3)
parser.add_argument("--num_sub_heads", type=int, default=5)

parser.add_argument("--out_root", type=str,
                    default="/scratch/shared/slow/xuji/iid_private")
parser.add_argument("--restart", default=False, action="store_true")

parser.add_argument("--save_freq", type=int, default=5)
parser.add_argument("--test_src", default=False, action="store_true")

parser.add_argument("--batchnorm_track", default=False, action="store_true")

# data transforms
parser.add_argument("--no_sobel", default=False, action="store_true")
parser.add_argument("--include_rgb", default=False, action="store_true")

parser.add_argument("--pre_scale_all", default=False,
                    action="store_true")
parser.add_argument("--pre_scale_factor", type=float, default=0.5)
parser.add_argument("--use_random_scale", default=False, action="store_true")
parser.add_argument("--scale_min", type=float, default=0.6)
parser.add_argument("--scale_max", type=float, default=1.4)
parser.add_argument("--input_sz", type=int, default=161)  # half of kazuto1011

# transforms we learn invariance to
parser.add_argument("--jitter_brightness", type=float, default=0.4)
parser.add_argument("--jitter_contrast", type=float, default=0.4)
parser.add_argument("--jitter_saturation", type=float, default=0.4)
parser.add_argument("--jitter_hue", type=float, default=0.125)

parser.add_argument("--flip_p", type=float, default=0.5)

parser.add_argument("--use_random_affine", default=False,
                    action="store_true")  # new
parser.add_argument("--aff_min_rot", type=float, default=-30.)  # degrees
parser.add_argument("--aff_max_rot", type=float, default=30.)  # degrees
parser.add_argument("--aff_min_shear", type=float, default=-10.)  # degrees
parser.add_argument("--aff_max_shear", type=float, default=10.)  # degrees
parser.add_argument("--aff_min_scale", type=float, default=0.8)
parser.add_argument("--aff_max_scale", type=float, default=1.2)

# local spatial invariance. Dense means done convolutionally. Sparse means done
#  once in data augmentation phase. These are not mutually exclusive
parser.add_argument("--half_T_side_dense", type=int, default=0)
parser.add_argument("--half_T_side_sparse_min", type=int, default=0)
parser.add_argument("--half_T_side_sparse_max", type=int, default=0)

config = parser.parse_args()

# Setup ------------------------------------------------------------------------

config.out_dir = os.path.join(config.out_root, str(config.model_ind))
config.dataloader_batch_sz = int(config.batch_sz / config.num_dataloaders)
assert (config.mode == "IID+")
assert (config.output_k >= config.gt_k)
assert (not (config.no_sobel and (not config.include_rgb)))
config.eval_mode = "orig"
config.use_doersch_datasets = False
set_segmentation_input_channels(config)

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

    # copy over new num_epochs and lr schedule
    config.num_epochs = given_config.num_epochs
    config.lr_schedule = given_config.lr_schedule
else:
    print("Given config: %s" % config_to_str(config))

# Model ------------------------------------------------------------------------

dataloaders, mapping_assignment_dataloader, mapping_test_dataloader = \
    segmentation_create_dataloaders(config)

net = archs.__dict__[config.arch](config)
if config.restart:
    dict = torch.load(os.path.join(config.out_dir, dict_name),
                      map_location=lambda storage, loc: storage)
    net.load_state_dict(dict["net"])
net.cuda()
net = torch.nn.DataParallel(net)
net.train()

optimiser = get_opt(config.opt)(net.module.parameters(), lr=config.lr)
if config.restart:
    optimiser.load_state_dict(dict["opt"])

# Results ----------------------------------------------------------------------

if config.restart:
    next_epoch = config.last_epoch + 1  # corresponds to last saved model
    print("starting from epoch %d" % next_epoch)

    config.epoch_acc = config.epoch_acc[:next_epoch]  # in case we overshot
    config.epoch_avg_subhead_acc = config.epoch_avg_subhead_acc[:next_epoch]
    config.epoch_stats = config.epoch_stats[:next_epoch]

    config.epoch_loss = config.epoch_loss[:(next_epoch - 1)]
    config.epoch_loss_no_lamb = config.epoch_loss_no_lamb[:(next_epoch - 1)]
else:
    config.epoch_acc = []
    config.epoch_avg_subhead_acc = []
    config.epoch_stats = []

    config.epoch_loss = []
    config.epoch_loss_no_lamb = []

    # torch.cuda.empty_cache()
    _ = segmentation_eval(config, net,
                          mapping_assignment_dataloader=mapping_assignment_dataloader,
                          mapping_test_dataloader=mapping_test_dataloader,
                          sobel=(not config.no_sobel),
                          using_IR=config.using_IR)

    print("Pre: time %s: \n %s" %
          (datetime.now(), nice(config.epoch_stats[-1])))
    sys.stdout.flush()
    next_epoch = 1

fig, axarr = plt.subplots(4, sharex=False, figsize=(20, 20))

if not config.use_uncollapsed_loss:
    print("using condensed loss (default)")
    loss_fn = IID_segmentation_loss
else:
    print("using uncollapsed loss")
    loss_fn = IID_segmentation_loss_uncollapsed

# Train ------------------------------------------------------------------------

for e_i in range(next_epoch, config.num_epochs):
    print("Starting e_i: %d %s" % (e_i, datetime.now()))
    sys.stdout.flush()

    iterators = (d for d in dataloaders)

    b_i = 0
    if e_i in config.lr_schedule:
        optimiser = update_lr(optimiser, lr_mult=config.lr_mult)

    avg_loss = 0.
    avg_loss_no_lamb = 0.
    avg_loss_count = 0

    for tup in zip(*iterators):
        net.module.zero_grad()

        if not config.no_sobel:
            # one less because this is before sobel
            pre_channels = config.in_channels - 1
        else:
            pre_channels = config.in_channels

        all_img1 = torch.zeros(config.batch_sz, pre_channels,
                               config.input_sz, config.input_sz).to(
            torch.float32).cuda()
        all_img2 = torch.zeros(config.batch_sz, pre_channels,
                               config.input_sz, config.input_sz).to(
            torch.float32).cuda()
        all_affine2_to_1 = torch.zeros(config.batch_sz, 2, 3).to(
            torch.float32).cuda()
        all_mask_img1 = torch.zeros(config.batch_sz, config.input_sz,
                                    config.input_sz).to(torch.float32).cuda()

        curr_batch_sz = tup[0][0].shape[0]
        for d_i in range(config.num_dataloaders):
            img1, img2, affine2_to_1, mask_img1 = tup[d_i]
            assert (img1.shape[0] == curr_batch_sz)

            actual_batch_start = d_i * curr_batch_sz
            actual_batch_end = actual_batch_start + curr_batch_sz

            all_img1[actual_batch_start:actual_batch_end, :, :, :] = img1
            all_img2[actual_batch_start:actual_batch_end, :, :, :] = img2
            all_affine2_to_1[actual_batch_start:actual_batch_end,
                             :, :] = affine2_to_1
            all_mask_img1[actual_batch_start:actual_batch_end,
                          :, :] = mask_img1

        if not (curr_batch_sz == config.dataloader_batch_sz) and (
                e_i == next_epoch):
            print("last batch sz %d" % curr_batch_sz)

        curr_total_batch_sz = curr_batch_sz * config.num_dataloaders
        all_img1 = all_img1[:curr_total_batch_sz, :, :, :]
        all_img2 = all_img2[:curr_total_batch_sz, :, :, :]
        all_affine2_to_1 = all_affine2_to_1[:curr_total_batch_sz, :, :]
        all_mask_img1 = all_mask_img1[:curr_total_batch_sz, :, :]

        if (not config.no_sobel):
            all_img1 = sobel_process(all_img1, config.include_rgb,
                                     using_IR=config.using_IR)
            all_img2 = sobel_process(all_img2, config.include_rgb,
                                     using_IR=config.using_IR)

        x1_outs = net(all_img1)
        x2_outs = net(all_img2)

        avg_loss_batch = None  # avg over the heads
        avg_loss_no_lamb_batch = None
        for i in range(config.num_sub_heads):
            loss, loss_no_lamb = loss_fn(x1_outs[i],
                                         x2_outs[i],
                                         all_affine2_to_1=all_affine2_to_1,
                                         all_mask_img1=all_mask_img1,
                                         lamb=config.lamb,
                                         half_T_side_dense=config.half_T_side_dense,
                                         half_T_side_sparse_min=config.half_T_side_sparse_min,
                                         half_T_side_sparse_max=config.half_T_side_sparse_max)

            if avg_loss_batch is None:
                avg_loss_batch = loss
                avg_loss_no_lamb_batch = loss_no_lamb
            else:
                avg_loss_batch += loss
                avg_loss_no_lamb_batch += loss_no_lamb

        avg_loss_batch /= config.num_sub_heads
        avg_loss_no_lamb_batch /= config.num_sub_heads

        if ((b_i % 100) == 0) or (e_i == next_epoch):
            print("Model ind %d epoch %d batch: %d avg loss %f avg loss no lamb %f "
                  "time %s" %
                  (config.model_ind, e_i, b_i, avg_loss_batch.item(),
                   avg_loss_no_lamb_batch.item(), datetime.now()))
            sys.stdout.flush()

        if not np.isfinite(float(avg_loss_batch.data)):
            print("Loss is not finite... %s:" % str(avg_loss_batch))
            exit(1)

        avg_loss += avg_loss_batch.item()
        avg_loss_no_lamb += avg_loss_no_lamb_batch.item()
        avg_loss_count += 1

        avg_loss_batch.backward()

        optimiser.step()

        b_i += 1
        if b_i == 2 and config.test_src:
            break

    # Eval -----------------------------------------------------------------------

    avg_loss = float(avg_loss / avg_loss_count)
    avg_loss_no_lamb = float(avg_loss_no_lamb / avg_loss_count)

    config.epoch_loss.append(avg_loss)
    config.epoch_loss_no_lamb.append(avg_loss_no_lamb)

    is_best = segmentation_eval(config, net,
                                mapping_assignment_dataloader=mapping_assignment_dataloader,
                                mapping_test_dataloader=mapping_test_dataloader,
                                sobel=(
                                    not config.no_sobel),
                                using_IR=config.using_IR)

    print("Pre: time %s: \n %s" %
          (datetime.now(), nice(config.epoch_stats[-1])))
    sys.stdout.flush()

    axarr[0].clear()
    axarr[0].plot(config.epoch_acc)
    axarr[0].set_title("acc (best), top: %f" % max(config.epoch_acc))

    axarr[1].clear()
    axarr[1].plot(config.epoch_avg_subhead_acc)
    axarr[1].set_title("acc (avg), top: %f" %
                       max(config.epoch_avg_subhead_acc))

    axarr[2].clear()
    axarr[2].plot(config.epoch_loss)
    axarr[2].set_title("Loss")

    axarr[3].clear()
    axarr[3].plot(config.epoch_loss_no_lamb)
    axarr[3].set_title("Loss no lamb")

    fig.canvas.draw_idle()
    fig.savefig(os.path.join(config.out_dir, "plots.png"))

    if is_best or (e_i % config.save_freq == 0):
        net.module.cpu()
        save_dict = {"net": net.module.state_dict(),
                     "optimiser": optimiser.state_dict()}

        if e_i % config.save_freq == 0:
            torch.save(save_dict, os.path.join(
                config.out_dir, "latest.pytorch"))
            config.last_epoch = e_i  # for last saved version

        if is_best:
            torch.save(save_dict, os.path.join(config.out_dir, "best.pytorch"))

            with open(os.path.join(config.out_dir, "best_config.pickle"),
                      'wb') as outfile:
                pickle.dump(config, outfile)

            with open(os.path.join(config.out_dir, "best_config.txt"),
                      "w") as text_file:
                text_file.write("%s" % config)

        net.module.cuda()

    with open(os.path.join(config.out_dir, "config.pickle"), 'wb') as outfile:
        pickle.dump(config, outfile)

    with open(os.path.join(config.out_dir, "config.txt"), "w") as text_file:
        text_file.write("%s" % config)

    if config.test_src:
        exit(0)
