import torch
from torchmetrics import Metric

class MR (Metric):
    """
        Miss rate metric implementation
        The number of scenarios where none of the forecasted trajectories are within 2.0 meters of ground truth according to endpoint error.
        we seek to reduce this metric.
    """
    def __init__(self, miss_threshold: float = 2.0) -> None:
        super().__init__()
        self.miss_threshold = miss_threshold
        self.add_state('sum', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('count', default=torch.tensor(0), dist_reduce_fx='sum')
    
    def update(self, pred_traj: torch.Tensor, gt_traj: torch.Tensor) -> None:
        """_summary_

        Args:
            pred_traj (torch.Tensor): Predicted trajectory [bs, pred_len, 2]
            gt_traj (torch.Tensor): GT trajectory [bs, pred_len, 2]
        """
        self.sum += (torch.norm(pred_traj[:, -1] - gt_traj[:, -1], p=2, dim=-1) > self.miss_threshold).sum()
        self.count += pred_traj.size(0)
    
    def compute (self) -> torch.Tensor:
        return self.sum / self.count