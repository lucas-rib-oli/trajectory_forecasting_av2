import argparse
import torch
from torch import nn
from torch.utils.data import DataLoader
from pathlib import Path
from av2.datasets.motion_forecasting import scenario_serialization, data_schema
from av2.datasets.motion_forecasting.data_schema import ArgoverseScenario, ObjectType
from datasets import Av2MotionForecastingDataset, collate_fn
from model.transtraj import TransTraj
from typing import Final
from collections import defaultdict
from colorama import Fore
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from rich.progress import track
from metrics import Av2Metrics
from configs import Config
from losses import ClosestL2Loss
from av_eval_forecasting import compute_forecasting_metrics
# ===================================================================================== #
# Configure constants
_OBS_DURATION_TIMESTEPS: Final[int] = 50 # The first 5 s of each scenario is denoted as the observed window
_PRED_DURATION_TIMESTEPS: Final[int] = 60 # The subsequent 6 s is denoted as the forecasted/predicted horizon.
_TOTAL_DURATION_TIMESTEPS: Final[int] = 110
NUM_CENTERLINE_INTERP_PTS: Final[int] = 10 # Polyline size
# Lane class dict
# Lane class dict
LANE_MARKTYPE_DICT = {
    "DASH_SOLID_YELLOW": 1,
    "DASH_SOLID_WHITE": 2,
    "DASHED_WHITE": 3,
    "DASHED_YELLOW": 4,
    "DOUBLE_SOLID_YELLOW": 5,
    "DOUBLE_SOLID_WHITE": 6,
    "DOUBLE_DASH_YELLOW": 7,
    "DOUBLE_DASH_WHITE": 8,
    "SOLID_YELLOW": 9,
    "SOLID_WHITE": 10,
    "SOLID_DASH_WHITE": 11,
    "SOLID_DASH_YELLOW": 12,
    "SOLID_BLUE": 13,
    "NONE": 14,
    "UNKNOWN": 15
}
# ===================================================================================== #
def str_to_bool(value):
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
parser = argparse.ArgumentParser()
parser.add_argument('--path_2_dataset', type=str, default='pickle_data', help='Path to the dataset')
parser.add_argument('--path_2_configuration', type=str, default='configs/config_files/transtraj_config.py', help='Path to the configuration')
parser.add_argument('--experiment_name', type=str, default='none', help='Experiment name')
args = parser.parse_args()
# ===================================================================================== #
def normalice_heading (angle):
    norm_angle = (angle + np.pi) % (2 * np.pi) - np.pi
    return norm_angle
# ===================================================================================== #

def generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    mask = (torch.triu(torch.ones((sz, sz), device=self.device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask
# ===================================================================================== #
class TransformerEvaluation ():
    def __init__(self, cfg, num_scenarios: int = 100) -> None:

        model_config = cfg.get('model')
        self.d_model = model_config['d_model']
        self.nhead = model_config['nhead']
        self.num_encoder_layers = model_config['N']
        self.dropout = model_config['dropout']
        self.dim_feedforward = model_config['dim_feedforward']
        self.num_queries = model_config['num_queries']
        self.pose_dim = model_config['pose_dim']
        self.out_feats_size = model_config['out_feats_size']
        self.future_size = model_config['future_size']
        self.subgraph_width = model_config['subgraph_width']
        self.num_subgraph_layers = model_config['num_subgraph_layers']
        self.lane_channels = model_config['lane_channels']
        
        train_config = cfg.get('train')
        self.device = train_config['device']
        self.num_epochs = train_config['num_epochs']
        self.batch_size = train_config['batch_size']
        self.num_workers = train_config['num_workers']
        self.resume_train = train_config['resume_train']
        
        config_data = cfg.get('data')
        self.save_path = config_data['path_2_save_weights']
        self.name_pickle = config_data['name_pickle']
        if args.experiment_name == 'none':
            main_exp_name = train_config['experiment_name']
        else:
            main_exp_name = args.experiment_name
        self.experiment_name = main_exp_name + "_d_model_" + str(self.d_model) + "_nhead_" + str(self.nhead) + "_N_" + str(self.num_encoder_layers) + "_dffs_" + str(self.dim_feedforward)  + "_lseq_" + str(self.future_size)
        # ---------------------------------------------------------------------------------------------------- #
        # Get the model
        self.model = TransTraj (pose_dim=self.pose_dim, out_feats_size=self.out_feats_size, num_queries=self.num_queries,
                                subgraph_width=self.subgraph_width, num_subgraph_layers=self.num_subgraph_layers, lane_channels=self.lane_channels,
                                future_size=self.future_size,
                                d_model=self.d_model, nhead=self.nhead, N=self.num_encoder_layers, dim_feedforward=self.dim_feedforward, dropout=self.dropout).to(self.device)
        # ---------------------------------------------------------------------------------------------------- #
        # Validation data
        self.val_data = Av2MotionForecastingDataset (dataset_dir=args.path_2_dataset, split='val', output_traj_size=self.future_size,
                                                     name_pickle=self.name_pickle)
        self.val_dataloader = DataLoader (self.val_data, batch_size=1, num_workers=self.num_workers, shuffle=False, collate_fn=collate_fn)
        # ---------------------------------------------------------------------------------------------------- #
        self.loss_fn = ClosestL2Loss()
        # ---------------------------------------------------------------------------------------------------- #
        print (Fore.CYAN + 'Device: ' + Fore.WHITE + self.device + Fore.RESET)
        print (Fore.CYAN + 'Experiment name: ' + Fore.WHITE + self.experiment_name + Fore.RESET)
        print (Fore.CYAN + 'Number of data: ' + Fore.WHITE + str(len(self.val_data)) + Fore.RESET)
        # Load the model
        self.load_model ('best')
        # ----------------------------------------------------------------------- #
        # Metris
        self.av2Metrics = Av2Metrics()
    # ---------------------------------------------------------------------------------------------------- #
    def load_model (self, name: str):
        file_name = os.path.join(self.save_path, f'{self.experiment_name}_{name}.pth')
        if os.path.exists(file_name):
            self.model.load_state_dict(torch.load(file_name)['model_state_dict'])
        else:
            print (Fore.RED + 'Eror to open .pth' + Fore.RESET)
            sys.exit(0)
    # ---------------------------------------------------------------------------------------------------- #
    def create_mask(self, src: torch.Tensor, tgt: torch.Tensor):
        src_seq_len = src.shape[1]
        tgt_seq_len = tgt.shape[1]
        batch_size = src.shape[0]
        
        tgt_mask = generate_square_subsequent_mask(tgt_seq_len, self.device)
        src_mask = torch.zeros((src_seq_len, src_seq_len),device=self.device).type(torch.bool)

        src_padding_mask = torch.zeros((batch_size, src_seq_len),device=self.device).type(torch.bool)
        tgt_padding_mask = torch.zeros((batch_size, tgt_seq_len),device=self.device).type(torch.bool)
        
        return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask
    # ---------------------------------------------------------------------------------------------------- #
    def evaluation(self):
        # Set network in eval mode
        self.model.eval()
        
        validation_losses = []
        
        minADE_metrics = []
        minFDE_metrics = []
        mr_metrics = []
        p_minADE_metrics = []
        p_minFDE_metrics = []
        p_MR_metrics = []
        brier_minADE_metrics = []
        brier_minFDE_metrics = []
        
        forecasted_trajectories = {}
        gt_trajectories = {}
        forecasted_probabilities = {}
        id_obs = 0
        for idx, data in enumerate(self.val_dataloader):
            # Set no requires grad
            with torch.no_grad():                
                historic_traj: torch.Tensor = data['historic']
                future_traj: torch.Tensor = data['future']
                offset_future_traj: torch.Tensor = data['offset_future']
                lanes: torch.Tensor = torch.cat ([data['lanes'][:,:,:,:2], data['lanes'][:,:,:,3:]],dim=-1)
                lanes_mask: torch.Tensor = data['lanes_mask']
                # Pass to device
                historic_traj = historic_traj.to(self.device) 
                future_traj = future_traj.to(self.device)
                offset_future_traj = offset_future_traj.to(self.device)
                lanes = lanes.to(self.device)
                lanes_mask = lanes_mask.to(self.device)
                # ----------------------------------------------------------------------- #
                # Get valid agents
                valid_agents_mask = torch.abs(future_traj).sum(dim=-1).sum(-1) > 0
                # ----------------------------------------------------------------------- #
                # Output model
                                   # x-7 ... x0 | x1 ... x7
                pred, conf = self.model (historic_traj, future_traj, lanes, src_mask=None, tgt_mask=None, src_padding_mask=None, tgt_padding_mask=None) # return -> x1 ... x7
                loss = self.loss_fn(pred, conf, future_traj, offset_future_traj)
                validation_losses.append(loss.detach().cpu().numpy())
                softmax = nn.Softmax(dim=-1)
                scores = softmax(conf)
                # ----------------------------------------------------------------------- #
                # Get trajectories
                for i, predicted_batch in enumerate(pred): # bs, a, k, f, d
                    for idx_agent, agent_predicted in enumerate(predicted_batch):# a, k, f, d
                        if valid_agents_mask[i,idx_agent]: 
                            forecasted_trajectories[id_obs] = []
                            gt_trajectories[id_obs] = future_traj[i][idx_agent][:,0:2].cpu().numpy()
                            forecasted_probabilities[id_obs] = scores[i][idx_agent].cpu().numpy()
                            for forescasted_traj in agent_predicted:
                                forecasted_trajectories[id_obs].append(forescasted_traj.cpu().numpy()[:,0:2])
                            id_obs += 1
        # ----------------------------------------------------------------------- #
        validation_loss = np.mean(validation_losses)
        # ----------------------------------------------------------------------- #
        metric_results = compute_forecasting_metrics(forecasted_trajectories, gt_trajectories, 6, 60, 2.0, forecasted_probabilities)
        for key, value in metric_results.items():
            print(Fore.GREEN + key + ': ' + Fore.WHITE, "{:.4f}".format(value))
        print (Fore.GREEN + 'Validation loss:' + Fore.WHITE, validation_loss)
        print (Fore.GREEN + 'Forecasted trajectories: ' + Fore.WHITE, len(forecasted_trajectories))
        # Compute metrics argoverse
        
    # ---------------------------------------------------------------------------------------------------- #
        
# ===================================================================================== #
def main ():
    cfg = Config.fromfile(args.path_2_configuration)
    
    # Model in prediction mode
    model_predict = TransformerEvaluation (cfg)
    # Train the model
    model_predict.evaluation()
    # model_predict.visualization_prediction()
# ===================================================================================== #
if __name__ == '__main__':
    if not torch.cuda.is_available():
        print (Fore.CYAN + "Status: " + Fore.RED + "cuda not available" + Fore.RESET)
        exit(-1)
    main()