import torch
from torch import nn
import numpy as np

class ClosestL2Loss (nn.Module):
    def __init__(self, reduction: str = 'sum') -> None:
        super().__init__()
        
        self.reg_loss_fn = nn.PairwiseDistance(p=2)
        self.class_loss_fn = nn.BCELoss()
    # ===================================================================================== #   
    def get_one_hot_vector_by_distance (self, pred_trajs: torch.Tensor, gt_trajs: torch.Tensor) -> torch.Tensor:
        # Get the distance between the prediction and the gt trajectories
        dist = torch.norm(gt_trajs[:,:,:,:2] - pred_trajs[:,:,:,:2], p=2, dim=-1)
        distance_sum = torch.sum(dist, dim=-1)
        indexes = torch.argmin(distance_sum,dim=-1)
        one_hot_vector = torch.zeros_like (distance_sum).to(pred_trajs.device)
        one_hot_vector = one_hot_vector.scatter(2, indexes.unsqueeze(-1), 1)
        return one_hot_vector, indexes    
    # ===================================================================================== #
    def closest_trajectory_loss (self, pred: torch.Tensor, gt_overdim: torch.Tensor):
        """Compute the regresion loss with respect to the closest predicted trajectory

        Args:
            pred (torch.Tensor): _description_
            gt (torch.Tensor): _description_
        """
        # import matplotlib.pyplot as plt
        # trajs_pred_0 = pred[0]
        # gt_0 = gt[0]
        # print ('closest_traj size: ', closest_traj.size())
        # for k in range(0,6):
        #     plt.plot (trajs_pred_0[k,:,0].detach().cpu().numpy(), trajs_pred_0[k,:,1].detach().cpu().numpy(), '--o', color='r', label='prediction')
        # plt.plot (gt_0[0,:,0].detach().cpu().numpy(), gt_0[0,:,1].detach().cpu().numpy(), '--o', color=(0,1,0), label='GT')
        # plt.plot (closest_traj[0,0,:,0].detach().cpu().numpy(), closest_traj[0,0,:,1].detach().cpu().numpy(), '--o', color=(1,1,0), label='best')
        # plt.show()
        
        # Compute loss
        reg_loss = self.reg_loss_fn(pred, gt_overdim)
        # Get the clostest distance
        min_reg_loss, _ = torch.min(torch.sum(reg_loss, dim=-1), dim=-1)
        # Sum agent loss
        reg_loss = torch.sum(min_reg_loss, dim=-1)    
        return reg_loss.mean()
    # ===================================================================================== #
    def forward (self, pred_trajs: torch.Tensor, pred_scores: torch.Tensor, gt_trajs: torch.Tensor, offset_gt_trajs: torch.Tensor) -> torch.Tensor:
        """Forward function

        Args:
            pred_trajs (torch.Tensor): _description_
            gt_trajs (torch.Tensor): _description_
            offset_gt_trajs (torch.Tensor): _description_
        """
        bs = pred_trajs.shape[0]
        K = pred_trajs.shape[2] # Number trajectory of a target
        
        future_traj_size = pred_trajs.shape[3]
        output_pose_dim = pred_trajs.shape[-1]
        # ----------------------------------------------------------------------- #
        gt_overdim: torch.Tensor = gt_trajs.unsqueeze(2).repeat(1, 1, K, 1, 1)
        # offset_gt_trajs_overdim: torch.Tensor = offset_gt_trajs.unsqueeze(1).repeat(1, K, 1, 1)
        # ----------------------------------------------------------------------- #
        # Compute classification loss
        one_hot_vector, _ = self.get_one_hot_vector_by_distance (pred_trajs, gt_overdim)
        cls_los = self.class_loss_fn(pred_scores, one_hot_vector)
        # ----------------------------------------------------------------------- #        
        # Compute regresion loss
        # Get the loss with respect the best prediction
        reg_loss = self.closest_trajectory_loss(pred_trajs, gt_overdim)
        # ----------------------------------------------------------------------- #
        # Final loss
        loss = reg_loss + cls_los
        
        return loss

class LaplaceNLLLoss(nn.Module):

    def __init__(self,
                 eps: float = 1e-6,
                 reduction: str = 'mean') -> None:
        super(LaplaceNLLLoss, self).__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self,
                pred: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        loc, scale = pred.chunk(2, dim=-1)
        scale = scale.clone()
        with torch.no_grad():
            scale.clamp_(min=self.eps)
        nll = torch.log(2 * scale) + torch.abs(target - loc) / scale
        if self.reduction == 'mean':
            return nll.mean()
        elif self.reduction == 'sum':
            return nll.sum()
        elif self.reduction == 'none':
            return nll
        else:
            raise ValueError('{} is not a valid value for reduction'.format(self.reduction))