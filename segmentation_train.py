"""
Train a diffusion model on images.
"""
import sys
import numpy as np
import argparse
#import torchsummary as summary
#from pytorch_model_summary import summary

sys.path.append("")
sys.path.append("scripts")
from guided_diffusion import dist_util, logger
from guided_diffusion.resample import create_named_schedule_sampler
#from guided_diffusion.bratsloader import BRATSDataset
from guided_diffusion.shaprloader import SHAPRDataset
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
import torch as th
from guided_diffusion.train_util import TrainLoop
#from visdom import Visdom
#viz = Visdom(port=8850)

def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(params)
    model.to(dist_util.dev())
    #print(model)
    #print(summary(model, input_size=(1, 2, 64, 64, 64)))
    #print(summary(model(th.zeros((1, 2, 64, 64, 64)), th.tensor([200,250,300,350]))))
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion,  maxt=1000)
    logger.log("creating data loader...")
    #ds = BRATSDataset(args.data_dir, test_flag=False)
    ds = SHAPRDataset(args.data_dir, test_flag=False)
    datal= th.utils.data.DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True)
    data = iter(datal)


    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        classifier=None,
        data=data,
        dataloader=datal,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
    ).run_loop()


def create_argparser():
    defaults = dict(
        data_dir="/home/dominik/Documents/SHAPR_diffusion/data_SHAPR/",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=4,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=100,
        save_interval=5000,
        resume_checkpoint='',#'"./results/pretrainedmodel.pt",
        use_fp16=True,
        fp16_scale_growth=1e-3,
        learn_sigma = True,
        class_cond = False,
        num_res_blocks = 1, #2,
        num_heads = 1,
        use_scale_shift_norm = False,
        rescale_learned_sigmas = True,
        attention_resolutions = 16, #16,
        diffusion_steps = 1000,
        image_size = 64,
        num_channels = 32,
        noise_schedule = "linear",
        rescale_timesteps = False
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser




if __name__ == "__main__":
    main()
