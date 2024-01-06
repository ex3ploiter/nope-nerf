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

    

    
    while epoch_it < (  scheduling_epoch):
        iteration=epoch_it
        gaussian_net.update_learning_rate(iteration)
        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussian_net.oneupSHdegree()
        bg = torch.rand((3), device="cuda") if opt.random_background else background

        loss=trainer.train_step_singleview(pipe=pipe,bg=bg)

        epoch_it+=1
        
        # gaussian_net.save_ply(os.path.join("./point_cloud.ply"))

        # with torch.no_grad():
           
        #     # Densification
        #     if iteration < opt.densify_until_iter:
        #         # Keep track of max radii in image-space for pruning
        #         gaussian_net.max_radii2D[:,visibility_filter] = torch.max(gaussian_net.max_radii2D[:,visibility_filter], radii[visibility_filter])
        #         gaussian_net.add_densification_stats(viewspace_point_tensor, visibility_filter)

        #         if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
        #             size_threshold = 20 if iteration > opt.opacity_reset_interval else None
        #             gaussian_net.densify_and_prune(opt.densify_grad_threshold, 0.005, scene_net.cameras_extent, size_threshold)
                
        #         if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
        #             gaussian_net.reset_opacity()

            
            
            # -------------
        del  loss,viewspace_point_tensor, visibility_filter, radii
        torch.cuda.empty_cache()

    
    
            
    epoch_it=0
    while epoch_it < (  scheduling_epoch):
        iteration=epoch_it
        gaussian_net.update_learning_rate(iteration)
        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussian_net.oneupSHdegree()
        bg = torch.rand((3), device="cuda") if opt.random_background else background

        trainer.train_step_3dgsTransform(pipe=pipe,bg=bg)

        epoch_it+=1
        
    
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
    