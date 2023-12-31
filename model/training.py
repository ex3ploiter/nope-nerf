import os
from random import randint
import torch
import logging
from model.losses import Loss
import numpy as np
from PIL import Image
import imageio
from torch.nn import functional as F
from model.common import (
    get_tensor_values, 
     arange_pixels,  project_to_cam, transform_to_world,
)
logger_py = logging.getLogger(__name__)
class Trainer(object):
    def __init__(self, optimizer, cfg, device=None, optimizer_pose=None, pose_param_net=None, 
                    optimizer_focal=None, focal_net=None, optimizer_distortion=None,distortion_net=None,gaussian_net=None,scene_net=None, **kwargs):
        """model trainer

        Args:
            model (nn.Module): model
            optimizer (optimizer):pytorch optimizer object
            cfg (dict): config argument options
            device (device): Pytorch device option. Defaults to None.
            optimizer_pose (optimizer, optional): pytorch optimizer for poses. Defaults to None.
            pose_param_net (nn.Module, optional): model with pose parameters. Defaults to None.
            optimizer_focal (optimizer, optional): pytorch optimizer for focal. Defaults to None.
            focal_net (nn.Module, optional): model with focal parameters. Defaults to None.
            optimizer_distortion (optimizer, optional): pytorch optimizer for depth distortion. Defaults to None.
            distortion_net (nn.Module, optional): model with distortion parameters. Defaults to None.
        """
        
        self.optimizer = optimizer
        self.device = device
        self.optimizer_pose = optimizer_pose
        self.pose_param_net = pose_param_net
        self.focal_net = focal_net
        self.optimizer_focal = optimizer_focal
        self.distortion_net = distortion_net
        self.optimizer_distortion = optimizer_distortion
        
        self.n_training_points = cfg['n_training_points']
        self.rendering_technique = cfg['type']
        self.vis_geo = cfg['vis_geo']

        self.detach_gt_depth = cfg['detach_gt_depth']
        self.pc_ratio = cfg['pc_ratio']
        self.match_method = cfg['match_method']
        self.shift_first = cfg['shift_first']
        self.detach_ref_img = cfg['detach_ref_img']
        self.scale_pcs = cfg['scale_pcs']
        self.detach_rgbs_scale = cfg['detach_rgbs_scale']
        self.vis_reprojection_every = cfg['vis_reprojection_every']
        self.nearest_limit = cfg['nearest_limit']
        self.annealing_epochs = cfg['annealing_epochs']

        self.pc_weight = cfg['pc_weight']
        self.rgb_s_weight = cfg['rgb_s_weight']
        self.rgb_weight = cfg['rgb_weight']
        self.depth_weight = cfg['depth_weight']
        self.weight_dist_2nd_loss = cfg['weight_dist_2nd_loss']
        self.weight_dist_1st_loss = cfg['weight_dist_1st_loss']
        self.depth_consistency_weight = cfg['depth_consistency_weight']


        self.loss = Loss(cfg)


        self.gaussian_net=gaussian_net
        self.scene_net=scene_net


    def train_step(self, data, it=None, epoch=None,scheduling_start=None, render_path=None):
        ''' Performs a training step.

        Args:
            data (dict): data dictionary
            it (int): training iteration
            epoch(int): current number of epochs
            scheduling_start(int): num of epochs to start scheduling
        '''
        self.model.train()
        self.optimizer.zero_grad()
        if self.pose_param_net:
           self.pose_param_net.train()
           self.optimizer_pose.zero_grad()
        if self.focal_net:
            self.focal_net.train()
            self.optimizer_focal.zero_grad()
        if self.distortion_net:
            self.distortion_net.train()
            self.optimizer_distortion.zero_grad()
        loss_dict = self.compute_loss(data, it=it, epoch=epoch, scheduling_start=scheduling_start, out_render_path=render_path)
        loss = loss_dict['loss']
        loss.backward()
        self.optimizer.step()
        if self.optimizer_pose:
            self.optimizer_pose.step()
        if self.optimizer_focal:
            self.optimizer_focal.step()
        if self.optimizer_distortion:
            self.optimizer_distortion.step()
        return loss_dict

    
    def render_visdata(self, data, resolution, it, out_render_path):
        (img, dpt, camera_mat, scale_mat, img_idx) = self.process_data_dict(data)
        h, w = resolution
        if self.pose_param_net:
            c2w = self.pose_param_net(img_idx)
            world_mat = torch.inverse(c2w).unsqueeze(0)
        if self.optimizer_focal:
            fxfy = self.focal_net(0)
            camera_mat = torch.tensor([[[fxfy[0], 0, 0, 0], 
                [0, -fxfy[1], 0, 0],
                [0, 0, -1, 0],
                [0, 0, 0, 1]]]).to(self.device)
        p_idx = torch.arange(h*w).to(self.device)
        p_loc, pixels = arange_pixels(resolution=(h, w))
        
        pixels = pixels.to(self.device)
        depth_input = dpt

        with torch.no_grad():
            rgb_pred = []
            depth_pred = []
            for i, (pixels_i, p_idx_i) in enumerate(zip(torch.split(pixels, 1024, dim=1), torch.split(p_idx, 1024, dim=0))):
                out_dict = self.model(
                     pixels_i, p_idx_i, camera_mat, world_mat, scale_mat, self.rendering_technique,
                    add_noise=False, eval_mode=True, it=it, depth_img=depth_input, img_size=(h, w))
                rgb_pred_i = out_dict['rgb']
                rgb_pred.append(rgb_pred_i)
                depth_pred_i = out_dict['depth_pred']
                depth_pred.append(depth_pred_i)
                
            rgb_pred = torch.cat(rgb_pred, dim=1)
            depth_pred = torch.cat(depth_pred, dim=0)
     
            rgb_pred = rgb_pred.view(h, w, 3).detach().cpu().numpy()
            img_out = (rgb_pred * 255).astype(np.uint8)
            depth_pred_out = depth_pred.view(h, w).detach().cpu().numpy()
            imageio.imwrite(os.path.join(out_render_path,'%04d_depth.png'% img_idx), 
            np.clip(255.0 / depth_pred_out.max() * (depth_pred_out - depth_pred_out.min()), 0, 255).astype(np.uint8))
            
            img1 = Image.fromarray(
                (img_out).astype(np.uint8)
            ).convert("RGB").save(
                os.path.join(out_render_path, '%04d_img.png' % img_idx)
            )
        if self.vis_geo:
            with torch.no_grad():
                rgb_pred = \
                    [self.model(
                         pixels_i, None, camera_mat, world_mat, scale_mat, 'phong_renderer',
                        add_noise=False, eval_mode=True, it=it, depth_img=depth_input, img_size=(h, w))['rgb']
                        for i, pixels_i in enumerate(torch.split(pixels, 1024, dim=1))]
            
                rgb_pred = torch.cat(rgb_pred, dim=1).cpu()              
                rgb_pred = rgb_pred.view(h, w, 3).detach().cpu().numpy()
                img_out = (rgb_pred * 255).astype(np.uint8)
                  
                
                img1 = Image.fromarray(
                    (img_out).astype(np.uint8)
                ).convert("RGB").save(
                    os.path.join(out_render_path, '%04d_geo.png' % img_idx)
                )

        return img_out.astype(np.uint8)
    def process_data_dict(self, data):
        ''' Processes the data dictionary and returns respective tensors
        Args:
            data (dictionary): data dictionary
        '''
        device = self.device
        img = data.get('img').to(device)
        img_idx = data.get('img.idx')
        dpt = data.get('img.dpt').to(device).unsqueeze(1)
        camera_mat = data.get('img.camera_mat').to(device)
        scale_mat = data.get('img.scale_mat').to(device)
       
        return (img, dpt, camera_mat, scale_mat, img_idx)
    def process_data_reference(self, data):
        ''' Processes the data dictionary and returns respective tensors
        Args:
            data (dictionary): data dictionary
        '''
        device = self.device
        ref_imgs = data.get('img.ref_imgs').to(device)
        ref_dpts = data.get('img.ref_dpts').to(device).unsqueeze(1)
        ref_idxs = data.get('img.ref_idxs')
        return ( ref_imgs, ref_dpts, ref_idxs)
    def anneal(self, start_weight, end_weight, anneal_start_epoch, anneal_epoches, current):
        """Anneal the weight from start_weight to end_weight
        """
        if current <= anneal_start_epoch:
            return start_weight
        elif current >= anneal_start_epoch + anneal_epoches:
            return end_weight
        else:
            return start_weight + (end_weight - start_weight) * (current - anneal_start_epoch) / anneal_epoches
        
    def compute_loss(self, data, eval_mode=False, it=None, epoch=None, scheduling_start=None, out_render_path=None):
        ''' Compute the loss.

        Args:
            data (dict): data dictionary
            eval_mode (bool): whether to use eval mode
            it (int): training iteration
            epoch(int): current number of epochs
            scheduling_start(int): num of epochs to start scheduling
            out_render_path(str): path to save rendered images
        '''
        
        viewpoint_stack = [scene.getTrainCameras().copy() for scene in self.scene_net]
        viewpoint_cam = [ view.pop(randint(0, len(viewpoint_stack)-1)) for view in viewpoint_stack]
        gt_images = [view.original_image.cuda() for view in viewpoint_cam]

        
        
        for idx in len(gt_images):

            render_pkg = render(viewpoint_cam[idx], gaussian_net[idx], pipe, bg)
            image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

            
            l1_loss(image, gt_image) 
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
