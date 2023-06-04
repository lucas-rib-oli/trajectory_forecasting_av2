import torch
import numpy as np

LOW_PROB_THRESHOLD_FOR_METRICS = 0.05

class Av2Metrics ():
    def __init__(self) -> None:
        pass
    
    def get_minFDE(self, forecasted_trajectories: torch.Tensor, gt_trajectory: torch.Tensor):
        """Compute Minimum Final Displacement Error

        Args:
            forecasted_trajectories (torch.Tensor): Forecasted trajectories [BS, K, F, D]
            gt_trajectory (torch.Tensor): GT trajectory [BS, F, D]
        """
        K = forecasted_trajectories.shape[1]
        gt_traj_overdim = gt_trajectory.unsqueeze(1).repeat(1, K, 1, 1)
        fde = torch.norm(forecasted_trajectories[:,:, -1, :2] - gt_traj_overdim[:,:, -1, :2], p=2, dim=-1)
        min_fde, _ = torch.min(fde, dim=-1)
        return min_fde
    
    def get_minADE(self, forecasted_trajectories: torch.Tensor, gt_trajectory: torch.Tensor):
        """Compute Minimum Average Displacement Error

        Args:
            forecasted_trajectories (torch.Tensor): Forecasted trajectories [BS, K, F, D]
            gt_trajectory (torch.Tensor): GT trajectory [BS, F, D]
        """
        K = forecasted_trajectories.shape[1]
        gt_traj_overdim = gt_trajectory.unsqueeze(1).repeat(1, K, 1, 1)
        ade = torch.norm(forecasted_trajectories[:,:, :, :2] - gt_traj_overdim[:,:, :, :2], p=2, dim=-1).mean(-1)
        min_ade, _ = torch.min(ade, dim=-1)
        return min_ade
    
    def get_MR(self, forecasted_trajectories: torch.Tensor, gt_trajectory: torch.Tensor, miss_threshold: float = 2.0):
        """Compute Miss Rate

        Args:
            forecasted_trajectories (torch.Tensor): Forecasted trajectories [BS, K, F, D]
            gt_trajectory (torch.Tensor): GT trajectory [BS, F, D]
        """
        K = forecasted_trajectories.shape[1]
        gt_traj_overdim = gt_trajectory.unsqueeze(1).repeat(1, K, 1, 1)
        fde = torch.norm(forecasted_trajectories[:,:, -1, :2] - gt_traj_overdim[:,:, -1, :2], p=2, dim=-1)
        min_fde, _ = torch.min(fde, dim=-1)
        mr = (min_fde > miss_threshold).float()
        return mr

    def get_p_minFDE(self, forecasted_trajectories: torch.Tensor, gt_trajectory: torch.Tensor, scores: torch.Tensor):
        """Compute Probabilistic Minimum Final Displacement Error

        Args:
            forecasted_trajectories (torch.Tensor): Forecasted trajectories [BS, K, F, D]
            gt_trajectory (torch.Tensor): GT trajectory [BS, F, D]
            scores (torch.Tensor): Score [BS, K]
        """
        K = forecasted_trajectories.shape[1]
        gt_traj_overdim = gt_trajectory.unsqueeze(1).repeat(1, K, 1, 1)
        
        fde = torch.norm(forecasted_trajectories[:,:, -1, :2] - gt_traj_overdim[:,:, -1, :2], p=2, dim=-1)
        min_fde, index_fde = torch.min(fde, dim=-1)
        
        scores_fde = torch.gather(scores, 1, index_fde.unsqueeze(1)).squeeze(-1)
        min_fde_scores = torch.min( -torch.log (scores_fde), -torch.log(torch.tensor(LOW_PROB_THRESHOLD_FOR_METRICS)) )
        p_minFDE = min_fde + min_fde_scores
        
        return p_minFDE
    
    def get_p_minADE(self, forecasted_trajectories: torch.Tensor, gt_trajectory: torch.Tensor, scores: torch.Tensor):
        """Compute Minimum Average Displacement Error

        Args:
            forecasted_trajectories (torch.Tensor): Forecasted trajectories [BS, K, F, D]
            gt_trajectory (torch.Tensor): GT trajectory [BS, F, D]
            scores (torch.Tensor): Score [BS, K]
        """
        K = forecasted_trajectories.shape[1]
        gt_traj_overdim = gt_trajectory.unsqueeze(1).repeat(1, K, 1, 1)
        
        fde = torch.norm(forecasted_trajectories[:,:, -1, :2] - gt_traj_overdim[:,:, -1, :2], p=2, dim=-1)
        min_fde, index_fde = torch.min(fde, dim=-1)
        
        ade = torch.norm(forecasted_trajectories[:,:, :, :2] - gt_traj_overdim[:,:, :, :2], p=2, dim=-1).mean(-1)
        min_ade, _ = torch.min(ade, dim=-1)
        
        scores_fde = torch.gather(scores, 1, index_fde.unsqueeze(1)).squeeze(-1)
        min_fde_scores = torch.min( -torch.log (scores_fde), -torch.log(torch.tensor(LOW_PROB_THRESHOLD_FOR_METRICS)) )
        p_minADE = min_ade + min_fde_scores
        
        return p_minADE
    
    def get_p_MR(self, forecasted_trajectories: torch.Tensor, gt_trajectory: torch.Tensor, scores: torch.Tensor, miss_threshold: float = 2.0):
        """Compute Probabilistic Miss Rate

        Args:
            forecasted_trajectories (torch.Tensor): Forecasted trajectories [BS, K, F, D]
            gt_trajectory (torch.Tensor): GT trajectory [BS, F, D]
            scores (torch.Tensor): Score [BS, K]
        """
        K = forecasted_trajectories.shape[1]
        gt_traj_overdim = gt_trajectory.unsqueeze(1).repeat(1, K, 1, 1)
        
        fde = torch.norm(forecasted_trajectories[:,:, -1, :2] - gt_traj_overdim[:,:, -1, :2], p=2, dim=-1)
        min_fde, index_fde = torch.min(fde, dim=-1)
        
        scores_fde = torch.gather(scores, 1, index_fde.unsqueeze(1)).squeeze(-1)
        
        mr = min_fde > miss_threshold
        p_mr = mr.to(torch.float32)
        for i in range(0, mr.shape[0]):
            if mr[i]:
                p_mr[i] = 1.0
            else:
               p_mr[i] = 1 - scores_fde[i]
            
        return p_mr
    
    def get_brier_minFDE(self, forecasted_trajectories: torch.Tensor, gt_trajectory: torch.Tensor, scores: torch.Tensor):
        """Compute Brier Minimum Final Displacement Error

        Args:
            forecasted_trajectories (torch.Tensor): Forecasted trajectories [BS, K, F, D]
            gt_trajectory (torch.Tensor): GT trajectory [BS, F, D]
            scores (torch.Tensor): Score [BS, K]
        """
        K = forecasted_trajectories.shape[1]
        gt_traj_overdim = gt_trajectory.unsqueeze(1).repeat(1, K, 1, 1)
        
        fde = torch.norm(forecasted_trajectories[:,:, -1, :2] - gt_traj_overdim[:,:, -1, :2], p=2, dim=-1)
        min_fde, index_fde = torch.min(fde, dim=-1)
        
        scores_fde = torch.gather(scores, 1, index_fde.unsqueeze(1)).squeeze(-1)
        brier_minFDE = min_fde + torch.pow ((torch.tensor(1.0) - scores_fde), 2)
        
        return brier_minFDE

    def get_brier_minADE(self, forecasted_trajectories: torch.Tensor, gt_trajectory: torch.Tensor, scores: torch.Tensor):
        """Compute Brier Average Displacement Error

        Args:
            forecasted_trajectories (torch.Tensor): Forecasted trajectories [BS, K, F, D]
            gt_trajectory (torch.Tensor): GT trajectory [BS, F, D]
            scores (torch.Tensor): Score [BS, K]
        """
        K = forecasted_trajectories.shape[1]
        gt_traj_overdim = gt_trajectory.unsqueeze(1).repeat(1, K, 1, 1)
        
        fde = torch.norm(forecasted_trajectories[:,:, -1, :2] - gt_traj_overdim[:,:, -1, :2], p=2, dim=-1)
        min_fde, index_fde = torch.min(fde, dim=-1)
        
        ade = torch.norm(forecasted_trajectories[:,:, :, :2] - gt_traj_overdim[:,:, :, :2], p=2, dim=-1).mean(-1)
        min_ade, _ = torch.min(ade, dim=-1)
        
        scores_fde = torch.gather(scores, 1, index_fde.unsqueeze(1)).squeeze(-1)
        brier_minADE = min_ade + torch.pow ((torch.tensor(1.0) - scores_fde), 2)
        
        return brier_minADE
    
    def get_metrics(self, forecasted_trajectories: torch.Tensor, gt_trajectory: torch.Tensor, scores: torch.Tensor):
        metric_results = {}
        metric_results["minADE"] = torch.mean(self.get_minADE(forecasted_trajectories, gt_trajectory))
        metric_results["minFDE"] = torch.mean(self.get_minFDE(forecasted_trajectories, gt_trajectory))
        metric_results["MR"] = torch.mean(self.get_MR(forecasted_trajectories, gt_trajectory))
        metric_results["p-minADE"] = torch.mean(self.get_p_minADE(forecasted_trajectories, gt_trajectory, scores))
        metric_results["p-minFDE"] = torch.mean(self.get_p_minFDE(forecasted_trajectories, gt_trajectory, scores))
        metric_results["p-MR"] = torch.mean(self.get_p_MR(forecasted_trajectories, gt_trajectory, scores))
        metric_results["brier-minADE"] = torch.mean(self.get_brier_minADE(forecasted_trajectories, gt_trajectory, scores))
        metric_results["brier-minFDE"] = torch.mean(self.get_brier_minFDE(forecasted_trajectories, gt_trajectory, scores))
        
        return metric_results
        
    