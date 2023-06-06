import torch
import numpy as np

LOW_PROB_THRESHOLD_FOR_METRICS = 0.05

class Av2Metrics ():
    def __init__(self) -> None:
        pass
    
    def get_minFDE(self, forecasted_trajectory: torch.Tensor, gt_trajectory: torch.Tensor):
        """Compute Minimum Final Displacement Error

        Args:
            forecasted_trajectories (torch.Tensor): Forecasted trajectories [BS, F, D]
            gt_trajectory (torch.Tensor): GT trajectory [BS, F, D]
        """
        fde = torch.norm(forecasted_trajectory[:,-1, :2] - gt_trajectory[:,-1, :2], p=2, dim=-1)
        return fde
    
    def get_minADE(self, forecasted_trajectory: torch.Tensor, gt_trajectory: torch.Tensor):
        """Compute Minimum Average Displacement Error

        Args:
            forecasted_trajectory (torch.Tensor): Forecasted trajectories [BS, F, D]
            gt_trajectory (torch.Tensor): GT trajectory [BS, F, D]
        """
        ade = torch.norm(forecasted_trajectory[:,:, :2] - gt_trajectory[:,:, :2], p=2, dim=-1).mean(-1)
        return ade
    
    def get_MR(self, forecasted_trajectory: torch.Tensor, gt_trajectory: torch.Tensor, miss_threshold: float = 2.0):
        """Compute Miss Rate

        Args:
            forecasted_trajectory (torch.Tensor): Forecasted trajectories [BS, F, D]
            gt_trajectory (torch.Tensor): GT trajectory [BS, F, D]
        """
        fde = torch.norm(forecasted_trajectory[:,-1, :2] - gt_trajectory[:,-1, :2], p=2, dim=-1)
        mr = (fde > miss_threshold).float()
        return mr

    def get_metrics(self, forecasted_trajectory: torch.Tensor, gt_trajectory: torch.Tensor):
        metric_results = {}
        metric_results["minADE"] = torch.mean(self.get_minADE(forecasted_trajectory, gt_trajectory))
        metric_results["minFDE"] = torch.mean(self.get_minFDE(forecasted_trajectory, gt_trajectory))
        metric_results["MR"] = torch.mean(self.get_MR(forecasted_trajectory, gt_trajectory))
        
        return metric_results
        
    