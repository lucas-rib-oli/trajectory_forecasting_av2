import torch

class minADE ():
    """
        Minimum Average Displacement Error (minADE)
        The average L2 distance between the best forecasted trajectory and the ground truth.
        The best here refers to the trajectory that has the minimum endpoint error
        We seek to reduce this metric.
    """
    def __init__(self) -> None:
        pass
    
    def compute(self, pred_traj: torch.Tensor, gt_traj: torch.Tensor) -> torch.Tensor:
        """_summary_

        Args:
            pred_traj (torch.Tensor): Predicted trajectory [bs, K, pred_len, 2]
            gt_traj (torch.Tensor): GT trajectory [bs, pred_len, 2]
        """
        num_traj = pred_traj.shape[1]
        gt_traj_overdim = gt_traj.unsqueeze(1).repeat(1, num_traj, 1, 1)
        ade = torch.norm(pred_traj - gt_traj_overdim, p=2, dim=-1).mean(-1)
        min_ade, _ = torch.min(ade, dim=-1)
        return torch.mean(min_ade)