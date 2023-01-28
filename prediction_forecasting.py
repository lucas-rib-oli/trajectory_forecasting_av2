import argparse
import torch
from torch import nn
from torch.utils.data import DataLoader
from pathlib import Path
from av2.datasets.motion_forecasting import scenario_serialization, data_schema
from av2.datasets.motion_forecasting.data_schema import ArgoverseScenario, ObjectType
from av2_motion_forecasting_dataset import Av2MotionForecastingDataset
from model.transformer import TransformerModel, generate_square_subsequent_mask
from typing import Final, Dict, List
from collections import defaultdict
from colorama import Fore
import os
import json
import sys
import numpy as np
import matplotlib.pyplot as plt
import threading
from rich.progress import track
from av2.utils.typing import NDArrayFloat, NDArrayInt
import numpy.typing as npt
# ===================================================================================== #
# Configure constants
_OBS_DURATION_TIMESTEPS: Final[int] = 8 # The first 5 s of each scenario is denoted as the observed window
_PRED_DURATION_TIMESTEPS: Final[int] = 12 # The subsequent 6 s is denoted as the forecasted/predicted horizon.
_TOTAL_DURATION_TIMESTEPS: Final[int] = 20
# ===================================================================================== #
def str_to_bool(value):
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
parser = argparse.ArgumentParser()
parser.add_argument('--path_2_dataset', type=str, default='/datasets/', help='Path to the dataset')
parser.add_argument('--path_2_configuration', type=str, default='config_transformer.json', help='Path to the configuration')
args = parser.parse_args()

class TransformerPrediction ():
    def __init__(self, config : dict, num_scenarios: int = 100) -> None:

        self.d_model = config['d_model']
        self.nhead = config['nhead']
        self.num_encoder_layers = config['num_encoder_layers']
        self.dim_feedforward = config['dim_feedforward']
        self.enc_inp_size = config['enc_inp_size']
        self.dec_inp_size = config['dec_inp_size']
        self.dec_out_size = config['dec_out_size']
        
        self.device = config['device']
        self.num_epochs = config['num_epochs']
        self.learning_rate = config['lr']
        self.current_lr = self.learning_rate
        self.batch_size = 2048
        self.output_traj_size = config['output_traj_size']
        self.num_workers = config['num_workers']
        self.dropout = config['dropout']
        self.save_path = 'models_weights/'
        self.experiment_name = config['experiment_name'] + "_d_model_" + str(self.d_model) + "_nhead_" + str(self.nhead) + "_N_" + str(self.num_encoder_layers) + "_dffs_" + str(self.dim_feedforward)  + "_lseq_" + str(self.output_traj_size)
        
        self.filename_pickle_src = config['filename_pickle_src']
        self.filename_pickle_tgt = config['filename_pickle_tgt']
        
        self.model = TransformerModel (enc_inp_size=self.enc_inp_size, dec_inp_size=self.dec_inp_size, dec_out_size=self.dec_out_size, 
                                       d_model=self.d_model, nhead=self.nhead, N=self.num_encoder_layers, dim_feedforward=self.dim_feedforward).to(self.device)
        # ---------------------------------------------------------------------------------------------------- #
        # self.val_data = Av2MotionForecastingDataset (dataset_dir=args.path_2_dataset, split='val', output_traj_size=self.output_traj_size, 
        #                                              load_pickle=False, save_pickle=False)
        # self.val_dataloader = DataLoader (self.val_data, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)
        # ---------------------------------------------------------------------------------------------------- #
        self.loss_fn = nn.HuberLoss(reduction='mean')
        # ---------------------------------------------------------------------------------------------------- #
        argoverse_scenario_dir = os.path.join(args.path_2_dataset, 'argoverse2', 'motion_forecasting', 'val')
        argoverse_scenario_dir = Path(argoverse_scenario_dir)
        all_scenario_files = sorted(argoverse_scenario_dir.rglob("*.parquet"))
        all_scenario_files = all_scenario_files[200:300]
        
        self.src_actor_trajectory_by_id: Dict[str, npt.NDArray] = {}
        self.tgt_actor_trajectory_by_id: Dict[str, npt.NDArray] = {}
        
        self.src_actor_trajectory_by_scenarios : Dict[str, npt.NDArray] = {}
        self.tgt_actor_trajectory_by_scenarios : Dict[str, npt.NDArray] = {}
        
        # ---------------------------------------------------------------------------------------------------- #
        idx = 0
        begin = 0
        end = 0
        threads = list()
        scenario_path_splitted = list()
        num_threads = 1
        length = int (len(all_scenario_files) / num_threads)
        remainder = int (len(all_scenario_files) % num_threads)
        
        for _ in range (0, min(len(all_scenario_files), num_threads)):
            if remainder > 0:
                if remainder == 0:
                    end += length
                else:
                    end += length + 1
                remainder -= 1
            else:
                end += length
            scenario_path_splitted.append (all_scenario_files[begin:end])
            begin = end
        
        for i in range(0, len(scenario_path_splitted)):
            x = threading.Thread(target=self.__generate_scenario_parallel, args=(scenario_path_splitted[i],))
            threads.append(x)
            x.start()
        # Join threads
        for index, thread in track(enumerate(threads)):
            thread.join()
        
        # for scenario_path in track(all_scenario_files):
        #     self.__generate_scenario(scenario_path)
        
        print (Fore.CYAN + 'Device: ' + Fore.WHITE + self.device + Fore.RESET)
        print (Fore.CYAN + 'Experiment name: ' + Fore.WHITE + self.experiment_name + Fore.RESET)
        # Load the model
        self.load_model ('best')
    # ---------------------------------------------------------------------------------------------------- #
    def __generate_scenario_parallel (self, scenarios_path: List) -> None:
        for scenario_path in scenarios_path:
            self.__generate_scenario(scenario_path)
    def __generate_scenario (self, scenario_path: Path) -> None:
        """_summary_

        Args:
            scenario_path (Path): Path to the parquet file corresponding to the Argoverse scenario
        """
        # ----------------------------------------------------------------------- #
        scenario_id = scenario_path.stem.split("_")[-1]
        static_map_path = scenario_path.parents[0] / f"log_map_archive_{scenario_id}.json"
        try:
            scenario = scenario_serialization.load_argoverse_scenario_parquet(scenario_path)
        except:
            print(Fore.RED + 'Fail to read: ' + Fore.RESET, scenario_path)
            return
        # static_map = ArgoverseStaticMap.from_json(static_map_path)
        
        scenario_src_actor_trajectory_by_id: Dict[str, npt.NDArray] = {}
        scenario_tgt_actor_trajectory_by_id: Dict[str, npt.NDArray] = {}
        
        # Get trajectories
        idx = 0
        for track in scenario.tracks:
            # Only get the vehicles and dont save the ego-vehicle (AV), and 
            if track.object_type != ObjectType.VEHICLE or track.track_id == "AV":
                continue
            # Only get the 'FOCAL TACK' and 'SCORED CARS'
            if track.category != data_schema.TrackCategory.FOCAL_TRACK and track.category != data_schema.TrackCategory.SCORED_TRACK:
                continue
            # Get timesteps for which actor data is valid
            actor_timesteps: NDArrayInt = np.array( [object_state.timestep for object_state in track.object_states] )

            if actor_timesteps.shape[0] < _TOTAL_DURATION_TIMESTEPS:
                continue
            # Get actor trajectory and heading history and instantaneous velocity
            actor_state: NDArrayFloat = np.array( [list(object_state.position) + [np.sin(object_state.heading), np.cos(object_state.heading)] + list(object_state.velocity) for object_state in track.object_states])

            # Get source actor trajectory and heading history -> observerd or historical trajectory
            src_actor_trajectory = actor_state[:_OBS_DURATION_TIMESTEPS]
            # Get target actor trajectory and heading history -> forescated or predicted trajectory
            tgt_actor_trajectory = actor_state[_OBS_DURATION_TIMESTEPS:_TOTAL_DURATION_TIMESTEPS]
            
            idx += 1
            scenario_src_actor_trajectory_by_id[track.track_id] = src_actor_trajectory
            scenario_tgt_actor_trajectory_by_id[track.track_id] = tgt_actor_trajectory
        
        if scenario.focal_track_id in scenario_tgt_actor_trajectory_by_id.keys():
            # Get the final observed trajectory of the focal agent
            focal_coordinate = scenario_src_actor_trajectory_by_id[scenario.focal_track_id][-1, 0:2]
                        
            src_full_traj = scenario_src_actor_trajectory_by_id[scenario.focal_track_id][:, 0:2]
            tgt_full_traj = scenario_tgt_actor_trajectory_by_id[scenario.focal_track_id][:, 0:2]
            
            heading_vector = scenario_src_actor_trajectory_by_id[scenario.focal_track_id][-1, 2:4]
            sin_heading = heading_vector[0]
            cos_heading = heading_vector[1]
            # Get heading
            focal_heading = np.arctan2(sin_heading,
                                       cos_heading)
            
            src_zeros_vector = np.zeros((src_full_traj.shape[0], 1))
            src_ones_vector = np.ones((src_full_traj.shape[0], 1))
            tgt_zeros_vector = np.zeros((tgt_full_traj.shape[0], 1))
            tgt_ones_vector = np.ones((tgt_full_traj.shape[0], 1))
            
            rot_matrix = np.array([[cos_heading, -sin_heading, 0, 0],
                                   [sin_heading,  cos_heading, 0, 0],
                                   [          0,            0, 1, 0],
                                   [          0,            0, 0, 1]])
              
            scenario_src_actor_trajectory_by_id_transformed = {}
            scenario_tgt_actor_trajectory_by_id_transformed = {}
            # Transform all trajectories
            for track_id in scenario_tgt_actor_trajectory_by_id.keys():
                src_agent_coordinate = scenario_src_actor_trajectory_by_id[track_id][:, 0:2]
                tgt_agent_coordinate = scenario_tgt_actor_trajectory_by_id[track_id][:, 0:2]
                
                src_agent_heading = np.arctan2(scenario_src_actor_trajectory_by_id[track_id][:,2], scenario_src_actor_trajectory_by_id[track_id][:,3])
                tgt_agent_heading = np.arctan2(scenario_tgt_actor_trajectory_by_id[track_id][:,2], scenario_tgt_actor_trajectory_by_id[track_id][:,3])
                
                src_agent_velocity = scenario_src_actor_trajectory_by_id[track_id][:, 4:]
                tgt_agent_velocity = scenario_tgt_actor_trajectory_by_id[track_id][:, 4:]
                
                # Add Z --> 0
                src_agent_coordinate = np.append (src_agent_coordinate, src_zeros_vector, axis=1)
                src_agent_coordinate = np.append (src_agent_coordinate, src_ones_vector, axis=1)
                tgt_agent_coordinate = np.append (tgt_agent_coordinate, tgt_zeros_vector, axis=1)
                tgt_agent_coordinate = np.append (tgt_agent_coordinate, tgt_ones_vector, axis=1)
                
                src_agent_velocity = np.append (src_agent_velocity, src_zeros_vector, axis=1)
                src_agent_velocity = np.append (src_agent_velocity, src_ones_vector, axis=1)
                tgt_agent_velocity = np.append (tgt_agent_velocity, tgt_zeros_vector, axis=1)
                tgt_agent_velocity = np.append (tgt_agent_velocity, tgt_ones_vector, axis=1)
                # Substract the center
                src_agent_coordinate = src_agent_coordinate - np.append (focal_coordinate, [0, 0])
                tgt_agent_coordinate = tgt_agent_coordinate - np.append (focal_coordinate, [0, 0])
                
                # Transformed trajectory
                src_agent_coordinate_tf = np.dot(rot_matrix, src_agent_coordinate.T).T
                tgt_agent_coordinate_tf = np.dot(rot_matrix, tgt_agent_coordinate.T).T
                # Transformed velocitys
                src_agent_velocity_tf = np.dot(rot_matrix, src_agent_velocity.T).T
                tgt_agent_velocity_tf = np.dot(rot_matrix, tgt_agent_velocity.T).T
                src_agent_velocity_tf = src_agent_velocity_tf[:,0:2] # Get only the components
                tgt_agent_velocity_tf = tgt_agent_velocity_tf[:,0:2]
                # Transformed heading
                src_agent_heading_tf = src_agent_heading - focal_heading
                tgt_agent_heading_tf = tgt_agent_heading - focal_heading
                # Normalice heading [-pi, pi)
                src_agent_heading_tf = (src_agent_heading_tf + np.pi) % (2 * np.pi) - np.pi
                tgt_agent_heading_tf = (tgt_agent_heading_tf + np.pi) % (2 * np.pi) - np.pi
                # Vector heading
                src_agent_vector_heading_tf = np.array([np.sin(src_agent_heading_tf), np.cos(src_agent_heading_tf)])
                tgt_agent_vector_heading_tf = np.array([np.sin(tgt_agent_heading_tf), np.cos(tgt_agent_heading_tf)])
                
                # Add heading
                src_agent_coordinate_tf[:,2:4] = src_agent_vector_heading_tf.T
                tgt_agent_coordinate_tf[:,2:4] = tgt_agent_vector_heading_tf.T
                
                # Add velocity
                src_agent_coordinate_tf = np.append (src_agent_coordinate_tf, src_agent_velocity_tf, axis=1)
                tgt_agent_coordinate_tf = np.append (tgt_agent_coordinate_tf, tgt_agent_velocity_tf, axis=1)
                
                # Save the trajectory
                self.src_actor_trajectory_by_id[track_id] = src_agent_coordinate_tf
                self.tgt_actor_trajectory_by_id[track_id] = tgt_agent_coordinate_tf
                
                scenario_src_actor_trajectory_by_id_transformed[track_id] = src_agent_coordinate_tf
                scenario_tgt_actor_trajectory_by_id_transformed[track_id] = tgt_agent_coordinate_tf
                
            self.src_actor_trajectory_by_scenarios[scenario_id] = scenario_src_actor_trajectory_by_id_transformed
            self.tgt_actor_trajectory_by_scenarios[scenario_id] = scenario_tgt_actor_trajectory_by_id_transformed
                
        else:
            # Not found focal agent or target agent
            # Delete scenario
            for track in scenario.tracks:
                if track.track_id in scenario_tgt_actor_trajectory_by_id.keys():
                    del scenario_src_actor_trajectory_by_id[track.track_id]
                    del scenario_tgt_actor_trajectory_by_id[track.track_id]
    # ---------------------------------------------------------------------------------------------------- #
    def load_model (self, name: str):
        file_name = os.path.join(self.save_path, f'{self.experiment_name}_{name}.pth')
        if os.path.exists(file_name):
            self.model.load_state_dict(torch.load(file_name)['model_state_dict'])
        else:
            print (Fore.RED + 'Eror to open .pth' + Fore.RESET)
            sys.exit(0)
    # ---------------------------------------------------------------------------------------------------- #
    def predict(self):
        # Set network in eval mode
        self.model.eval()
        for scenario_id in self.src_actor_trajectory_by_scenarios.keys():
            for track_id in self.src_actor_trajectory_by_scenarios[scenario_id].keys():
                with torch.no_grad():
                    src = torch.tensor(self.src_actor_trajectory_by_id[track_id], dtype=torch.float32).unsqueeze(0).to(self.device)
                    tgt = torch.tensor(self.tgt_actor_trajectory_by_id[track_id], dtype=torch.float32).unsqueeze(0).to(self.device)
                    # Get the start of the sequence
                    dec_inp = tgt[:, 0, :]
                    dec_inp = dec_inp.unsqueeze(1)
                    # Generate a square mask for the sequence
                    src_mask = torch.zeros(src.size()[1], src.size()[1]).to(self.device)
                    # ----------------------------------------------------------------------- #
                    # Encode step
                    memory = self.model.encode(src, src_mask).to(self.device)
                    # Apply Greedy Code
                    for _ in range (0, self.output_traj_size - 1):
                        # Get target mask
                        tgt_mask = (generate_square_subsequent_mask(dec_inp.size()[1])).to(self.device)
                        # Get tokens
                        out = self.model.decode(dec_inp, memory, tgt_mask).to(self.device)
                        # Generate the prediction
                        prediction = self.model.generate ( out ).to(self.device)
                        # Concatenate
                        dec_inp = torch.cat([dec_inp, prediction[:, -1:, :]], dim=1).to(self.device)
                    
                    plt.plot (dec_inp[0,:,0].cpu().numpy(), dec_inp[0,:,1].cpu().numpy(), color=(1,0,0), label='prediction' + str(track_id))
                    plt.plot (src[0,:,0].cpu().numpy(), src[0,:,1].cpu().numpy(), color=(0,0,1), label='historical' + str(track_id))
                    
                    plt.plot (tgt[0,:,0].cpu().numpy(), tgt[0,:,1].cpu().numpy(), color=(0,1,0), label='GT' + str(track_id))
                    plt.xlabel('X')
                    plt.ylabel('Y')
                    # plt.legend(loc="upper left")
            plt.show()
                # exit(0)
    # ---------------------------------------------------------------------------------------------------- #
    def compute_loss_val (self):
        # Set the network in evaluation mode
        self.model.eval()
        
        validation_losses = []
        for idx, data in track(enumerate(self.val_dataloader)):
            # Set no requires grad
            with torch.no_grad():
                src: torch.Tensor = data['src']
                tgt: torch.Tensor = data['tgt']
                src = src.to(self.device)
                # src = src.double()
                tgt = tgt.to(self.device)
                # tgt = tgt.double()
                
                # Generate a square mask for the sequence
                src_mask = torch.zeros(src.size()[1], src.size()[1]).to(self.device)
                
                # Get the start of the sequence
                # dec_inp = torch.zeros()
                # dec_inp = torch.zeros(tgt.shape[0], 1, tgt.shape[2]).to(self.device).double()
                dec_inp = tgt[:, 0, :]
                # Implement one dimension for the tranformer to be able to deal with the input decoder
                dec_inp = dec_inp.unsqueeze(1).to(self.device)
                # ----------------------------------------------------------------------- #
                # Encode step
                memory = self.model.encode(src, src_mask).to(self.device)
                # ----------------------------------------------------------------------- #
            
                # Apply Greedy Code
                for _ in range (0, self.output_traj_size - 1):
                    # Get target mask
                    tgt_mask = (nn.Transformer.generate_square_subsequent_mask(dec_inp.size()[1])).to(self.device)
                    # Get tokens
                    out = self.model.decode(dec_inp, memory, tgt_mask).to(self.device)
                    # Generate the prediction
                    prediction = self.model.generate ( out ).to(self.device)
                    # Concatenate
                    dec_inp = torch.cat([dec_inp, prediction[:, -1:, :]], dim=1).to(self.device)
                loss = self.loss_fn (dec_inp, tgt)
                loss = loss.mean()
                validation_losses.append(loss.detach().cpu().numpy())
        validation_loss = np.mean(validation_losses)
        print ('loss val: ', validation_loss)
        
                
            
# ===================================================================================== #
def main ():
    with open(args.path_2_configuration, "r") as config_file:
        config = json.load(config_file)
    
    # Model in prediction mode
    model_predict = TransformerPrediction (config)
    # Train the model
    model_predict.predict()
    # model_predict.visualization_prediction()
# ===================================================================================== #
if __name__ == '__main__':
    if not torch.cuda.is_available():
        print (Fore.CYAN + "Status: " + Fore.RED + "cuda not available" + Fore.RESET)
        exit(-1)
    main()