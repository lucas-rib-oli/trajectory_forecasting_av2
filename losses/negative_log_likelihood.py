import torch
from torch import nn

class NLLLoss (nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        self.log_std_range = (-1.609, 5.0)
        self.rho_limit = 0.5
        self.use_square_gmm = False
        self.class_loss_fn = nn.CrossEntropyLoss(reduction='none')
    # ===================================================================================== #
    def get_one_hot_vector_by_distance (self, pred_trajs: torch.Tensor, gt_trajs: torch.Tensor) -> torch.Tensor:
        # Get the distance between the prediction and the gt trajectories
        dist = torch.norm(gt_trajs[:,:,:,:,:2] - pred_trajs[:,:,:,:,:2], p=2, dim=-1)
        distance_sum = torch.sum(dist, dim=-1) # [BS, A, K]
        indexes = torch.argmin(distance_sum,dim=-1)
        one_hot_vector = torch.zeros_like (distance_sum).to(pred_trajs.device)
        one_hot_vector = one_hot_vector.scatter(2, indexes.unsqueeze(-1), 1)
        return one_hot_vector, indexes
    # ===================================================================================== #
    def forward (self, pred_trajs: torch.Tensor, pred_scores: torch.Tensor, gt_trajs: torch.Tensor, timestamp_loss_weight: torch.Tensor = None) -> torch.Tensor:
        """_summary_

        Args:
            pred_trajs (torch.Tensor): Predicted trajectories [BS, A, K, F, D] | D = (µ_x , µ_y , sigma_x , sigma_y , p)
            pred_scores (torch.Tensor): GT trajectories [BS, A, F, D]
            gt_trajs (torch.Tensor): _description_

        Returns:
            torch.Tensor: _description_
        """
        batch_size = pred_scores.shape[0]
        A = pred_trajs.shape[1]
        K = pred_trajs.shape[2]
        F = pred_trajs.shape[3]
        D = pred_trajs.shape[4]
        # ----------------------------------------------------------------------- #
        # Get valid agents
        valid_agents_mask = torch.abs(gt_trajs).sum(dim=-1).sum(dim=-1) > 0 # [BS, A]
        # ----------------------------------------------------------------------- #
        gt_overdim: torch.Tensor = gt_trajs.unsqueeze(2).repeat(1, 1, K, 1, 1)
        # ----------------------------------------------------------------------- #
        distance = torch.norm(pred_trajs[:,:,:,:, 0:2] - gt_overdim[:, :, :, :, 0:2], dim=-1).sum(-1)
        
        nearest_mode_idxs = torch.argmin (distance, dim=-1)
        nearest_traj_indexes = nearest_mode_idxs.view(batch_size, -1, 1, 1,1).repeat(1,1,1, F, D) 
        nearest_trajs = torch.gather(pred_trajs, 2, nearest_traj_indexes).squeeze(2)  # (BS, A, F, D)
        
        res_trajs = gt_trajs[:,:,:, 0:2] - nearest_trajs[:,:,:, 0:2]  # (BS, F, 2)
        dx = res_trajs[:,:,:, 0]
        dy = res_trajs[:,:,:, 1]

        if self.use_square_gmm:
            log_std1 = log_std2 = torch.clamp(nearest_trajs[:,:,:, 2], min=self.log_std_range[0], max=self.log_std_range[1])
            std1 = std2 = torch.exp(log_std1)   # (0.2m to 150m)
            rho = torch.zeros_like(log_std1)
        else:
            log_std1 = torch.clamp(nearest_trajs[:,:,:, 2], min=self.log_std_range[0], max=self.log_std_range[1])
            log_std2 = torch.clamp(nearest_trajs[:,:,:, 3], min=self.log_std_range[0], max=self.log_std_range[1])
            std1 = torch.exp(log_std1)  # (0.2m to 150m)
            std2 = torch.exp(log_std2)  # (0.2m to 150m)
            rho = torch.clamp(nearest_trajs[:,:,:, 4], min=-self.rho_limit, max=self.rho_limit)

        valid_agents_mask = valid_agents_mask.float()
        if timestamp_loss_weight is not None:
            valid_agents_mask = valid_agents_mask * timestamp_loss_weight[None, :]

        # -log(a^-1 * e^b) = log(a) - b
        reg_gmm_log_coefficient = log_std1 + log_std2 + 0.5 * torch.log(1 - rho**2)  # (BS, F)
        reg_gmm_exp = (0.5 * 1 / (1 - rho**2)) * ((dx**2) / (std1**2) + (dy**2) / (std2**2) - 2 * rho * dx * dy / (std1 * std2))  # (BS, F)

        reg_loss = ((reg_gmm_log_coefficient + reg_gmm_exp).sum(dim=-1) * valid_agents_mask) # [BS, A]
        # Sum agent loss
        reg_loss = reg_loss.sum(-1) # [BS]
        # Mean
        reg_loss = torch.mean(reg_loss)
        # ----------------------------------------------------------------------- # 
        # Compute classification loss
        one_hot_vector, _ = self.get_one_hot_vector_by_distance (pred_trajs, gt_overdim)
        pred_scores = pred_scores.permute(0, 2, 1)
        one_hot_vector = one_hot_vector.permute(0, 2, 1)
        cls_loss = self.class_loss_fn(pred_scores, one_hot_vector) # [BS, A]
        mean_cls_loss_masked = (cls_loss * valid_agents_mask).sum(-1) # [BS]
        final_cls_loss = torch.mean(mean_cls_loss_masked)
        # ----------------------------------------------------------------------- #   
        return reg_loss + final_cls_loss, reg_loss, final_cls_loss