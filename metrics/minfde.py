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
            pred_traj (torch.Tensor): Predicted trajectory [bs, pred_len, 2]
            gt_traj (torch.Tensor): GT trajectory [bs, pred_len, 2]
        """
        fde = torch.norm(pred_traj[:, -1, 0:2] - gt_traj[:,-1, 0:2], p=2, dim=-1)
        return torch.mean(fde)