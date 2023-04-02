import torch
from torchmetrics import Metric

class minFDE (Metric):
    """
        Minimum Final Displacement Error (minFDE)
        The L2 distance between the endpoint of the best forecasted trajectory and the ground truth. 
        The best here refers to the trajectory that has the minimum endpoint error.
        We seek to reduce this metric.
    """
    def __init__(self) -> None:
        super().__init__()
        self.add_state('sum', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('count', default=torch.tensor(0), dist_reduce_fx='sum')
    
    def update(self, pred_traj: torch.Tensor, gt_traj: torch.Tensor) -> None:
        """_summary_

        Args:
            pred_traj (torch.Tensor): Predicted trajectory [bs, pred_len, 2]
            gt_traj (torch.Tensor): GT trajectory [bs, pred_len, 2]
        """
        self.sum += torch.norm(pred_traj[:, -1] - gt_traj[:, -1], p=2, dim=-1).sum()
        self.count += pred_traj.size(0)
    
    def compute (self) -> torch.Tensor:
        return self.sum / self.count