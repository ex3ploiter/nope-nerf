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
    logger_py = logging.getLogger(__name__)

    # # Fix seeds
    np.random.seed(42)
    torch.manual_seed(42)
    
    is_cuda = (torch.cuda.is_available())
    device = torch.device("cuda" if is_cuda else "cpu")

    # params
    out_dir = cfg['training']['out_dir']
    backup_every = cfg['training']['backup_every']
    
    lr = cfg['training']['learning_rate']

    mode = cfg['training']['mode']
    train_loader, train_dataset = dl.get_dataloader(cfg, mode=mode, shuffle=cfg['dataloading']['shuffle'])
    test_loader, _ = dl.get_dataloader(cfg, mode=mode, shuffle=cfg['dataloading']['shuffle'])
    iter_test = iter(test_loader)
    data_test = next(iter_test)
    

    n_views = train_dataset['img'].N_imgs
    # init network
    network_type = cfg['model']['network_type']
    auto_scheduler = cfg['training']['auto_scheduler']
    scheduling_epoch = cfg['training']['scheduling_epoch']
    

    if network_type=='official':
        model = mdl.OfficialStaticNerf(cfg)
    
     # init renderer 
    rendering_cfg = cfg['rendering']
    renderer = mdl.Renderer(model, rendering_cfg, device=device)
    # init model
    nope_nerf = mdl.get_model(renderer, cfg, device=device)
    # init optimizer
    weight_decay = cfg['training']['weight_decay']
    optimizer = optim.Adam(nope_nerf.parameters(), lr=lr, weight_decay=weight_decay)

    # init checkpoints and load
    checkpoint_io = mdl.CheckpointIO(out_dir, model=nope_nerf, optimizer=optimizer)
    load_dir = cfg['training']['load_dir']

    try:
        load_dict = checkpoint_io.load(load_dir, load_model_only=cfg['training']['load_ckpt_model_only'])
    except FileExistsError:
        load_dict = dict()
        
    # resume training
    epoch_it = load_dict.get('epoch_it', -1)
    it = load_dict.get('it', -1)
    metric_val_best = load_dict.get(
    'loss_val_best', -np.inf)
    patient_count = load_dict.get('patient_count', 0)
    scheduling_start = load_dict.get('scheduling_start', cfg['training']['scheduling_start'])

    
    

    
    # init distortion parameters
    if cfg['gaussian']['learn_gaussian']:
        # distortion_net = mdl.Learn_Distortion(n_views, cfg['distortion']['learn_scale'], cfg['distortion']['learn_shift'], cfg).to(device=device)
        gaussian_net=GaussianModel(dataset.sh_degree)
        scene_net=Scene(dataset, gaussian_net)
        gaussian_net.training_setup(opt)
        
        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        

        # optimizer_distortion = optim.Adam(distortion_net.parameters(), lr=cfg['training']['distortion_lr'])
        # optimizer_guassian = optim.Adam(params=gaussian_net.parameters(), lr=cfg['training']['distortion_lr'])

        

        epoch_it = load_dict.get('epoch_it', -1)
        # if not auto_scheduler:
        #     scheduler_distortion = torch.optim.lr_scheduler.MultiStepLR(optimizer_guassian, 
        #                                                             milestones=list(range(scheduling_start, scheduling_epoch+scheduling_start, 100)),
                                                                    # gamma=cfg['training']['scheduler_gamma_distortion'], last_epoch=epoch_it)


   
   
     # init training
    training_cfg = cfg['training']
    trainer = mdl.Trainer(model=gaussian_net,optimizer=gaussian_net.optimizer, cfg=training_cfg, device=device , cfg_all=cfg
                        ,scene_net=scene_net)

    
    

    logger = SummaryWriter(os.path.join(out_dir, 'logs'))
        
    # init training output
    print_every = cfg['training']['print_every']
    checkpoint_every = cfg['training']['checkpoint_every']
    visualize_every = cfg['training']['visualize_every']
    validate_every = cfg['training']['validate_every']
    eval_pose_every = cfg['training']['eval_pose_every']
    eval_img_every = cfg['training']['eval_img_every']

    render_path = os.path.join(out_dir, 'rendering')
    if not os.path.exists(render_path):
        os.makedirs(render_path)
    


    # Print model
    nparameters = sum(p.numel() for p in nope_nerf.parameters())
    logger_py.info(nope_nerf)
    logger_py.info('Total number of parameters: %d' % nparameters)
    t0b = time.time()

    
    patient = cfg['training']['patient']
    length_smooth=cfg['training']['length_smooth']
    scheduling_mode = cfg['training']['scheduling_mode']
    psnr_window = []

    # torch.autograd.set_detect_anomaly(True)

    log_scale_shift_per_view = cfg['training']['log_scale_shift_per_view']
    scale_dict = {}
    shift_dict = {}
    # load gt poses for evaluation
    if eval_pose_every>0:
        gt_poses = train_dataset['img'].c2ws.to(device) 
    # for epoch_it in tqdm(range(epoch_start+1, exit_after), desc='epochs'):
    
    
    while epoch_it < (scheduling_start + scheduling_epoch):
        iteration=epoch_it
        gaussian_net.update_learning_rate(iteration)
        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussian_net.oneupSHdegree()
        bg = torch.rand((3), device="cuda") if opt.random_background else background

        trainer.train_step_singleview(pipe=pipe,bg=bg)

        epoch_it+=1

        # with torch.no_grad():
           

          
            # # Densification
            # if iteration < opt.densify_until_iter:
            #     # Keep track of max radii in image-space for pruning
            #     gaussian_net.max_radii2D[visibility_filter] = torch.max(gaussian_net.max_radii2D[visibility_filter], radii[visibility_filter])
            #     gaussian_net.add_densification_stats(viewspace_point_tensor, visibility_filter)

            #     if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
            #         size_threshold = 20 if iteration > opt.opacity_reset_interval else None
            #         gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                
            #     if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
            #         gaussians.reset_opacity()

            
            
            # -------------

    
    local_rot=deepcopy(gaussian_net._rotation)
    local_trans=deepcopy(gaussian_net._scaling)
            
    epoch_it=0
    while epoch_it < (scheduling_start + scheduling_epoch):
        iteration=epoch_it
        gaussian_net.update_learning_rate(iteration)
        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussian_net.oneupSHdegree()
        bg = torch.rand((3), device="cuda") if opt.random_background else background

        trainer.train_step_singleview(local_rot, local_trans,pipe=pipe,bg=bg)

        epoch_it+=1
            

      
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
    