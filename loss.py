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
        return one_hot_vector
        
    # ===================================================================================== #
    def forward (self, pred_trajs: torch.Tensor, pred_scores: torch.Tensor, gt_trajs: torch.Tensor) -> torch.Tensor:
        """Forward function

        Args:
            pred_trajs (torch.Tensor): _description_
            gt_trajs (torch.Tensor): _description_
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