import torch
from torchmetrics import Metric

class minFDE ():
    """
        Minimum Final Displacement Error (minFDE)
        The L2 distance between the endpoint of the best forecasted trajectory and the ground truth. 
        The best here refers to the trajectory that has the minimum endpoint error.
        We seek to reduce this metric.
    """
    def __init__(self) -> None:
        pass
        
    def compute(self, pred_traj: torch.Tensor, gt_traj: torch.Tensor) -> torch.Tensor:
        """_summary_

        Args:
            pred_traj (torch.Tensor): Predicted trajectory [BS, A, K, F, out_feats]
            gt_traj (torch.Tensor): GT trajectory [BS, A, F, out_feats]
        """
        num_traj = pred_traj.shape[2]
        gt_traj_overdim = gt_traj.unsqueeze(2).repeat(1, 1, num_traj, 1, 1)
        fde = torch.norm(pred_traj[:,:,:, -1, 0:2] - gt_traj_overdim[:,:,:, -1, 0:2], p=2, dim=-1)
        min_fde, _ = torch.min(fde, dim=-1)
        min_fde = torch.mean(min_fde)
        return torch.mean(min_fde)