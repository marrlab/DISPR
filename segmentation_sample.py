"""

Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os
import nibabel as nib
from visdom import Visdom
from skimage.io import imsave
viz = Visdom(port=8850)
import sys
import random
sys.path.append("scripts")
import numpy as np
import time
import torch as th
import torch.distributed as dist
from guided_diffusion import dist_util, logger
#from guided_diffusion.bratsloader import BRATSDataset
from guided_diffusion.shaprloader import SHAPRDataset

from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
seed=10
th.manual_seed(seed)
th.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)

def visualize(img):
    _min = img.min()
    _max = img.max()
    normalized_img = (img - _min)/ (_max - _min)
    return normalized_img

def dice_score(pred, targs):
    pred = (pred>0).float()
    return 2. * (pred*targs).sum() / (pred+targs).sum()


def main():
    args = create_argparser().parse_args()
    print(args)
    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )

    ds = SHAPRDataset(args.data_dir, test_flag=True)
    datal = th.utils.data.DataLoader(
        ds,
        batch_size=1,
        shuffle=False)
    data = iter(datal)
    all_images = []
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()
    while len(all_images) * args.batch_size < args.num_samples:
        b, path = next(data)  #should return an image from the dataloader "data"
        c = th.randn_like(b[:, :1, ...])
        print("c tensor size", c.shape)
        img = th.cat((b, c), dim=1)     #add a noise channel$
        print("img tensor size", img.shape)
        slice_ID = path
        logger.log("sampling...")
        start = th.cuda.Event(enable_timing=True)
        end = th.cuda.Event(enable_timing=True)
        for i in range(args.num_ensemble):  #this is for the generation of an ensemble of 5 masks.
            model_kwargs = {}
            start.record()
            sample_fn = (
                diffusion.p_sample_loop_known if not args.use_ddim else diffusion.ddim_sample_loop_known
            )
            sample, x_noisy, org = sample_fn(
                model,
                (args.batch_size, 3, args.image_size, args.image_size), img,
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
            )

            end.record()
            th.cuda.synchronize()
            print('time for 1 sample', start.elapsed_time(end))  #time measurement for the generation of 1 sample
            print("x_noisy sample", x_noisy.shape)
            s = th.tensor(sample)
            print("shape sample", sample.shape)
            print("s sample", s.shape)
            print("saving", './predictions/'+str(slice_ID)[2:-3])
            imsave('./predictions/'+str(slice_ID)[2:-3]+"_"+str(i)+".tif", s.cpu().detach().numpy())

def create_argparser():
    defaults = dict(
        data_dir="./data_SHAPR", #./data/testing",
        clip_denoised=True,
        num_samples=5,
        batch_size=1,
        use_ddim=False,
        model_path = "./results/savedmodel045000.pt",
        image_size = 64,
        num_channels = 32,
        num_res_blocks = 2,
        num_heads = 1,
        #model_path="",
        num_ensemble=1      #number of samples in the ensemble
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":

    main()
