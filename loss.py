import torch
from torch import nn
import numpy as np
class TransLoss (nn.Module):
    def __init__(self, reduction: str = 'sum') -> None:
        super().__init__()
        
        self.reg_loss_fn = nn.HuberLoss(reduction=reduction)
        self.class_loss_fn = nn.CrossEntropyLoss()
    # ===================================================================================== #   
    def get_one_hot_vector_by_distance (self, pred_trajs: torch.Tensor, gt_trajs: torch.Tensor) -> torch.Tensor:
        # Get the distance between the prediction and the gt trajectories
        squared_dis = torch.sum(torch.pow(pred_trajs[:,:,:,0:2] - gt_trajs[:,:,:,0:2], 2), dim=-1)
        distance_summ = torch.sum(squared_dis, dim=-1)
        val, index = torch.min(distance_summ,dim=-1)
        one_hot_vector = torch.zeros_like (distance_summ).to(pred_trajs.device)
        row_indices = np.arange(distance_summ.shape[0])
        one_hot_vector[row_indices, index] += torch.ones(len(row_indices)).to(pred_trajs.device)
        return one_hot_vector, index    
    # ===================================================================================== #
    def closest_trajectory_loss (self, pred: torch.Tensor, gt: torch.Tensor):
        """Compute the regresion loss with with respect to the closest predicted trajectory

        Args:
            pred (torch.Tensor): _description_
            gt (torch.Tensor): _description_
        """
        bs = pred.shape[0]
        # Compute distance
        dist = torch.mean(torch.abs(gt - pred), dim=(2, 3))
        # finde the closest trajectory to the gt 
        min_dist, min_idx = torch.min(dist, dim=1)
        # Get the closest trajectory
        closest_traj = torch.gather(pred, 1, min_idx.view(bs, 1, 1, 1).repeat(1, 1, 60, 2))
        # Compute loss
        reg_loss = self.reg_loss_fn(closest_traj, gt)
        return reg_loss
    # ===================================================================================== #
    def forward (self, pred_trajs: torch.Tensor, pred_scores: torch.Tensor, gt_trajs: torch.Tensor, offset_gt_trajs: torch.Tensor) -> torch.Tensor:
        """Forward function

        Args:
            pred_trajs (torch.Tensor): _description_
            gt_trajs (torch.Tensor): _description_
            offset_gt_trajs (torch.Tensor): _description_
        """
        bs = pred_trajs.shape[0]
        future_traj_size = pred_trajs.shape[2]
        output_pose_dim = pred_trajs.shape[-1]
        K = pred_trajs.shape[1] # Number trajectory of a target
        
        gt_overdim: torch.Tensor = gt_trajs.view(bs, 1, future_traj_size, output_pose_dim).repeat(1, K, 1, 1)
        # ----------------------------------------------------------------------- #        
        # Compute regresion loss
        reg_loss = self.reg_loss_fn(pred_trajs, gt_overdim)
        # ----------------------------------------------------------------------- #
        # Compute classification loss
        one_hot_vector = self.get_one_hot_vector_by_distance (pred_trajs, gt_overdim)
        cls_los = self.class_loss_fn(pred_scores, one_hot_vector)
        # ----------------------------------------------------------------------- #
        # Final loss
        loss = reg_loss + cls_los

        return loss