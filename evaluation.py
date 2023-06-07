import argparse
import torch
from torch import nn
from torch.utils.data import DataLoader
from pathlib import Path
from av2.datasets.motion_forecasting import scenario_serialization, data_schema
from av2.datasets.motion_forecasting.data_schema import ArgoverseScenario, ObjectType
from datasets import Av2MotionForecastingDataset
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
from losses import SingleL2Loss
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
parser.add_argument('--path_2_dataset', type=str, default='/raid/datasets/argoverse2/pickle_data', help='Path to the dataset')
parser.add_argument('--path_2_configuration', type=str, default='configs/config_files/transtraj_config.py', help='Path to the configuration')
args = parser.parse_args()
# ===================================================================================== #
def normalice_heading (angle):
    norm_angle = (angle + np.pi) % (2 * np.pi) - np.pi
    return norm_angle

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
        self.dec_out_size = model_config['dec_out_size']
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
        self.experiment_name = train_config['experiment_name'] + "_d_model_" + str(self.d_model) + "_nhead_" + str(self.nhead) + "_N_" + str(self.num_encoder_layers) + "_dffs_" + str(self.dim_feedforward)  + "_lseq_" + str(self.future_size)
        # ---------------------------------------------------------------------------------------------------- #
        # Get the model
        self.model = TransTraj (pose_dim=self.pose_dim, d_model=self.d_model, nhead=self.nhead, 
                                N=self.num_encoder_layers, dim_feedforward=self.dim_feedforward, dropout=self.dropout).to(self.device)
        # ---------------------------------------------------------------------------------------------------- #
        # Validation data
        self.val_data = Av2MotionForecastingDataset (dataset_dir=args.path_2_dataset, split='val', output_traj_size=self.future_size,
                                                     name_pickle=self.name_pickle)
        self.val_dataloader = DataLoader (self.val_data, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
        # ---------------------------------------------------------------------------------------------------- #
        self.loss_fn = SingleL2Loss()
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
    def generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """Generates an upper-triangular matrix of -inf, with zeros on diag."""
        mask = (torch.triu(torch.ones((sz, sz), device=self.device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    def create_mask(self, src: torch.Tensor, tgt: torch.Tensor):
        src_seq_len = src.shape[1]
        tgt_seq_len = tgt.shape[1]
        batch_size = src.shape[0]
        
        tgt_mask = self.generate_square_subsequent_mask(tgt_seq_len)
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
        
        for idx, data in track(enumerate(self.val_dataloader), 'Evaluating the model ...'):
            # Set no requires grad
            with torch.no_grad():                
                historic_traj: torch.Tensor = data['historic'].squeeze()
                future_traj: torch.Tensor = data['future'].squeeze()
                offset_future_traj: torch.Tensor = data['offset_future'].squeeze()
                # Pass to device
                historic_traj = historic_traj.to(self.device) 
                future_traj = future_traj.to(self.device)
                offset_future_traj = offset_future_traj.to(self.device)
                # ----------------------------------------------------------------------- #
                # Generate a square mask for the sequence
                src_seq_len = historic_traj.size()[1]
                src_mask = torch.zeros((src_seq_len, src_seq_len),device=self.device).type(torch.bool)
                # ----------------------------------------------------------------------- #
                # Encode step
                memory = self.model.encode(historic_traj, src_mask).to(self.device)
                # ----------------------------------------------------------------------- #
                dec_inp = future_traj[:, 0, :]
                # Implement one dimension for the tranformer to be able to deal with the input decoder
                dec_inp = dec_inp.unsqueeze(1).to(self.device)
                # ----------------------------------------------------------------------- #
                # Apply Greedy Code
                for _ in range (0, self.future_size - 1):
                    # Get target mask
                    tgt_mask = self.generate_square_subsequent_mask(dec_inp.size()[1])
                    # Get tokens
                    out = self.model.decode(dec_inp, memory, tgt_mask).to(self.device)
                    # Generate the prediction
                    prediction = self.model.generate ( out ).to(self.device)
                    # Concatenate
                    dec_inp = torch.cat([dec_inp, prediction[:, -1:, :]], dim=1).to(self.device)
                tgt_expected = future_traj[:, 1:]
                pred = dec_inp[:, 1:]
                loss = self.loss_fn(pred, tgt_expected)
                
                validation_losses.append(loss.detach().cpu().numpy())
                # ----------------------------------------------------------------------- #
                # Compute metrics
                # get the best agent --> The best here refers to the trajectory that has the minimum endpoint error
                
                av2_metrics_dict = self.av2Metrics.get_metrics(pred[:,:,:2],  tgt_expected[:,:,:2])
                
                minADE = av2_metrics_dict["minADE"]
                minFDE = av2_metrics_dict["minFDE"]
                MR = av2_metrics_dict["MR"]
                
                minADE_metrics.append (minADE.detach().cpu().numpy())
                minFDE_metrics.append (minFDE.detach().cpu().numpy())
                mr_metrics.append (MR.detach().cpu().numpy())
        validation_loss = np.mean(validation_losses)
        
        print (Fore.GREEN + 'Validation loss:' + Fore.WHITE, validation_loss)
        print(Fore.GREEN + 'minFDE:' + Fore.WHITE, "{:.4f}".format(np.mean(minFDE_metrics)))
        print(Fore.GREEN + 'minADE:' + Fore.WHITE, "{:.4f}".format(np.mean(minADE_metrics)))
        print(Fore.GREEN + 'MR:' + Fore.WHITE, "{:.4f}".format(np.mean(mr_metrics)))
        
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