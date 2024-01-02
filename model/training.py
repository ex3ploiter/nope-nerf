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


from gaussian_renderer import render,render_transform
import torch
from torchvision import transforms
from PIL import Image
from utils.loss_utils import l1_loss, ssim


from scipy.stats import multivariate_normal
from scipy.spatial.transform import Rotation

logger_py = logging.getLogger(__name__)


def transform_gaussian(mean, covariance, translation, rotation_quaternion):
    rotation = Rotation.from_quat(rotation_quaternion)
    rotation2 = Rotation.from_quat(rotation_quaternion)
    rotation2_transpose=Rotation.from_matrix(rotation2.as_matrix().T)
    mean  = rotation.apply(mean) + translation
    covariance= rotation2_transpose.apply(rotation2.apply(covariance)+translation)+translation
    return mean,covariance

def transform_vector(vector, translation, rotation_quaternion):
    rotation = Rotation.from_quat(rotation_quaternion)
    
    vector  = rotation.apply(vector) + translation
    
    return vector



class Trainer(object):
    def __init__(self, model,optimizer, cfg, device=None, 
                   scene_net=None, **kwargs):
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
        
        
        self.model=model
        self.gaussian_net=self.model
        self.optimizer = optimizer
        self.device = device
        
        
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


        
        self.scene_net=scene_net


    def train_step_singleview(self,pipe=None,bg=None):

        self.optimizer.zero_grad()
        loss=self.compute_loss_singleview( pipe=pipe,bg=bg)
        return loss


    def train_step_3dgsTransform(self,local_rot, local_scale,pipe=None,bg=None,optimizer_rot=None,optimizer_trans=None):

        self.optimizer_rot=optimizer_rot
        self.optimizer_trans=optimizer_trans
        
        self.optimizer.zero_grad()
        loss=self.compute_loss_3dgsTransform( local_rot, local_scale,pipe=pipe,bg=bg)
        return loss


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
        
    def compute_loss_3dgsTransform(self,local_rot, local_trans,pipe=None,bg=None):
        viewpoint_stack = self.scene_net.getTrainCameras().copy()
        # viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        
        
        loss_total=0

        for idx in range(len(viewpoint_stack)-1):
            self.optimizer_rot.zero_grad()
            self.optimizer_trans.zero_grad()
            
            

            # Cam1=viewpoint_stack[idx]
            Cam2=viewpoint_stack[idx+1]
            gt_image2 = Cam2.original_image.cuda()
            
            render_pkg = render_transform(Cam2, self.gaussian_net, pipe, bg,idx=idx,rot=local_rot[idx],trans=local_trans[idx])
            image, _, _, _ = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
            
            
            Ll1=l1_loss(image, gt_image2) 
            loss = (1.0 - 0.2) * Ll1 + 0.2 * (1.0 - ssim(image, gt_image2))

            loss.backward()
            self.optimizer_rot.step()
            self.optimizer_trans.step()

            loss_total+=loss.detach().item()
            
            torch.cuda.empty_cache()
            
            del render_pkg,Ll1,image,Cam2
            
            
           
        
        print("loss_total: ", loss_total)
        return loss_total
  
        
    
    def compute_loss_singleview(self,pipe=None,bg=None):
    
        viewpoint_stack = self.scene_net.getTrainCameras().copy()
        # viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        
        


        # pil_image = transforms.ToPILImage()(gt_image)
        # pil_image.save('/content/output_image.png')
        
        loss_total=0

        for idx in range(len(viewpoint_stack)):
            self.optimizer.zero_grad()

            Cam1=viewpoint_stack[idx]
            gt_image = Cam1.original_image.cuda()
            
            render_pkg = render(Cam1, self.gaussian_net, pipe, bg,idx=idx)
            image, _, _, _ = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
            
            
            transforms.ToPILImage()(gt_image).save('output_image1.png')
            transforms.ToPILImage()(image).save('output_image2.png')

            Ll1=l1_loss(image, gt_image) 
            loss = (1.0 - 0.2) * Ll1 + 0.2 * (1.0 - ssim(image, gt_image))

            loss.backward()
            self.optimizer.step()

            loss_total+=loss
        
        print("loss_total: ", loss_total)
        return loss_total



            




        
