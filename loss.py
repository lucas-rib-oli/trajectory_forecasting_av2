import torch
from torch import nn
import numpy as np

class TransLoss (nn.Module):
    def __init__(self, reduction: str = 'sum') -> None:
        super().__init__()
        
        self.reg_loss_fn = nn.PairwiseDistance(p=2)
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
        # ----------------------------------------------------------------------- #
        gt_overdim: torch.Tensor = gt_trajs.unsqueeze(1).repeat(1, K, 1, 1)
        # offset_gt_trajs_overdim: torch.Tensor = offset_gt_trajs.unsqueeze(1).repeat(1, K, 1, 1)
        # ----------------------------------------------------------------------- #
        # Compute classification loss
        one_hot_vector, index = self.get_one_hot_vector_by_distance (pred_trajs, gt_overdim)
        cls_los = self.class_loss_fn(pred_scores, one_hot_vector)
        # ----------------------------------------------------------------------- #        
        # Compute regresion loss
        # pred_trajs_offset = torch.vstack((pred_trajs[:,:,0,:].unsqueeze(2), pred_trajs[:,:,1:,:] - pred_trajs[:,:,:-1,:]))
        # pred_trajs_offset = torch.cat((pred_trajs[:,:,0,:].unsqueeze(2), pred_trajs[:,:,1:,:] - pred_trajs[:,:,:-1,:]), dim=2)
        # Get the loss with respect the best prediction
        reg_loss = self.closest_trajectory_loss(pred_trajs[:,:,:,:2], gt_overdim[:,:,:,:2])
        
        # reg_loss = self.reg_loss_fn (pred_trajs, gt_overdim[:,:,:,:2])
        # reduce the loss if is neccesary
        if len(reg_loss.shape) > 2:
           reg_loss = torch.sum(reg_loss, -1)
           reg_loss = torch.mean(reg_loss)
        elif len(reg_loss.shape) > 1:
            reg_loss = torch.mean(reg_loss)        
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