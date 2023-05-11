import torch
from torch import nn
import numpy as np

class SingleL2Loss (nn.Module):
    def __init__(self, reduction: str = 'sum') -> None:
        super().__init__()        
        self.reg_loss_fn = nn.PairwiseDistance(p=2)
    
    # ===================================================================================== #
    def forward (self, pred_trajs: torch.Tensor, gt_trajs: torch.Tensor) -> torch.Tensor:
        """Forward function

        Args:
            pred_trajs (torch.Tensor): _description_
            gt_trajs (torch.Tensor): _description_
            offset_gt_trajs (torch.Tensor): _description_
        """
        # ----------------------------------------------------------------------- #        
        # Compute regresion loss
        reg_loss = self.reg_loss_fn (pred_trajs, gt_trajs)
        # Reduce the loss if is neccesary
        if len(reg_loss.shape) > 2:
           reg_loss = torch.sum(reg_loss, -1)
           reg_loss = torch.mean(reg_loss)
        elif len(reg_loss.shape) > 1:
            reg_loss = torch.mean(reg_loss)        
        # ----------------------------------------------------------------------- #
        # Final loss
        loss = reg_loss
        return loss
