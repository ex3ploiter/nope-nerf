import os
import sys
import logging
import time
import argparse

import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import dataloading as dl
import model as mdl
from utils_poses.comp_ate import compute_ATE, compute_rpe
from model.common import backup,  mse2psnr
from utils_poses.align_traj import align_ate_c2b_use_a2b


from scene import Scene, GaussianModel
from arguments import ModelParams, PipelineParams, OptimizationParams
from argparse import ArgumentParser, Namespace
from copy import deepcopy




def train(cfg,dataset, opt, pipe):
    

    # # Fix seeds
    np.random.seed(42)
    torch.manual_seed(42)
    
    is_cuda = (torch.cuda.is_available())
    device = torch.device("cuda" if is_cuda else "cpu")

    # params
    
    
    
    

    
    
    
    scheduling_epoch = cfg['training']['scheduling_epoch']
    

    
    


        
    # resume training
    epoch_it = 0
    
    
    

    
    # init distortion parameters
    if cfg['gaussian']['learn_gaussian']:
        # distortion_net = mdl.Learn_Distortion(n_views, cfg['distortion']['learn_scale'], cfg['distortion']['learn_shift'], cfg).to(device=device)
        gaussian_net=GaussianModel(dataset.sh_degree)
        scene_net=Scene(dataset, gaussian_net)
        gaussian_net.training_setup(opt)
        
        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        
        

       
     

   
   
     # init training
    training_cfg = cfg['training']
    trainer = mdl.Trainer(model=gaussian_net,optimizer=gaussian_net.optimizer,optimizer_Pose=gaussian_net.optimizer_Pose, cfg=training_cfg, device=device , cfg_all=cfg
                        ,scene_net=scene_net)

    
    
    local_3dgs_epoch=50
    param_epoch=100
    
    
    progress_bar = tqdm(range(epoch_it, local_3dgs_epoch), desc="Training progress Local 3DGS")

    gaussian_net.save_ply_byFrames('./plyResults/initial_poly/')
    while epoch_it < local_3dgs_epoch :
        iteration=epoch_it
        gaussian_net.update_learning_rate(iteration)
        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussian_net.oneupSHdegree()
        bg = torch.rand((3), device="cuda") if opt.random_background else background

        loss=trainer.train_step_singleview(pipe=pipe,bg=bg)
        progress_bar.set_postfix({"Loss": f"{loss:.{7}f}"})
        progress_bar.update(1)


        epoch_it+=1
        
 
        del  loss
        torch.cuda.empty_cache()

    
    
    gaussian_net.save_ply_byFrames('./plyResults/local_poly/')     
    epoch_it=0
    progress_bar_param = tqdm(range(epoch_it, param_epoch), desc="Training progress Parameters 3DGS")
    
    while epoch_it < param_epoch:
        iteration=epoch_it
        gaussian_net.update_learning_rate(iteration)
        # Every 1000 its we increase the levels of SH up to a maximum degree
        bg = torch.rand((3), device="cuda") if opt.random_background else background
        loss=trainer.train_step_3dgsTransform(pipe=pipe,bg=bg)

        epoch_it+=1
        
        progress_bar_param.set_postfix({"Loss": f"{loss:.{7}f}"})
        progress_bar_param.update(1)
        
    
    gaussian_net.save_transrot()
    
    
            

      
if __name__=='__main__':
    # Arguments
    parser = argparse.ArgumentParser(
        description='Training of nope-nerf model'
    )

    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)

    parser.add_argument('config', type=str, help='Path to config file.')
    args = parser.parse_args()
    cfg = dl.load_config(args.config, 'configs/default.yaml')
    # backup model
    backup(cfg['training']['out_dir'], args.config)
    train(cfg=cfg,dataset=lp.extract(args),opt=op.extract(args),pipe=pp.extract(args))
    