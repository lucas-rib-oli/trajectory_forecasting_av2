import torch

class MR ( ):
    """
        Miss rate metric implementation
        The number of scenarios where none of the forecasted trajectories are within 2.0 meters of ground truth according to endpoint error.
        we seek to reduce this metric.
    """
    def __init__(self, miss_threshold: float = 2.0) -> None:
        super().__init__()
        self.miss_threshold = miss_threshold
    
    def compute(self, pred_traj: torch.Tensor, gt_traj: torch.Tensor) -> torch.Tensor:
        """_summary_

        Args:
            pred_traj (torch.Tensor): Predicted trajectory [bs, pred_len, 2]
            gt_traj (torch.Tensor): GT trajectory [bs, pred_len, 2]
        """
        mr = (torch.norm(pred_traj[:,:, -1] - gt_traj[:,:, -1], p=2, dim=-1) > self.miss_threshold).sum(-1).float()
        return torch.mean(mr)