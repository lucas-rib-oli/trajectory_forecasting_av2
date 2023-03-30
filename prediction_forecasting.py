import argparse
import torch
from torch import nn
from torch.utils.data import DataLoader
from pathlib import Path
from av2.datasets.motion_forecasting import scenario_serialization, data_schema
from av2.datasets.motion_forecasting.data_schema import ArgoverseScenario, ObjectType
from av2.map.map_api import ArgoverseStaticMap
from av2.geometry.interpolate import interp_arc
from datasets import Av2MotionForecastingDataset, collate_fn
from model.transtraj import TransTraj
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
import rich
from av2.utils.typing import NDArrayFloat, NDArrayInt
import numpy.typing as npt
from configs import Config
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
parser.add_argument(
    '--root-path',
    type=str,
    default='/datasets/argoverse2/',
    help='specify the root path of dataset')
parser.add_argument('--path_2_configuration', type=str, default='configs/config_files/transtraj_config.py', help='Path to the configuration')
args = parser.parse_args()

class TransformerPrediction ():
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
        self.save_path = 'models_weights/'
                
        self.save_path = 'models_weights/'
        self.experiment_name = train_config['experiment_name'] + "_d_model_" + str(self.d_model) + "_nhead_" + str(self.nhead) + "_N_" + str(self.num_encoder_layers) + "_dffs_" + str(self.dim_feedforward)  + "_lseq_" + str(self.future_size)
        # ---------------------------------------------------------------------------------------------------- #
        # Get the model
        self.model = TransTraj (pose_dim=self.pose_dim, dec_out_size=self.dec_out_size, num_queries=self.num_queries,
                                subgraph_width=self.subgraph_width, num_subgraph_layers=self.num_subgraph_layers, lane_channels=self.lane_channels,
                                future_size=self.future_size,
                                d_model=self.d_model, nhead=self.nhead, N=self.num_encoder_layers, dim_feedforward=self.dim_feedforward, dropout=self.dropout).to(self.device)
        # ---------------------------------------------------------------------------------------------------- #
        # self.val_data = Av2MotionForecastingDataset (dataset_dir=args.path_2_dataset, split='val', output_traj_size=self.output_traj_size, 
        #                                              load_pickle=False, save_pickle=False)
        # self.val_dataloader = DataLoader (self.val_data, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)
        # ---------------------------------------------------------------------------------------------------- #
        self.loss_fn = nn.HuberLoss(reduction='mean')
        # ---------------------------------------------------------------------------------------------------- #
        self.all_traj_data, self.all_map_data = self.prepare_data_av2(split='val')
        
        print (Fore.CYAN + 'Device: ' + Fore.WHITE + self.device + Fore.RESET)
        print (Fore.CYAN + 'Experiment name: ' + Fore.WHITE + self.experiment_name + Fore.RESET)
        # Load the model
        self.load_model ('check')
    # ---------------------------------------------------------------------------------------------------- #
    def prepare_data_av2 (self, split: str, num_scenerarios: int = 100):
        argoverse_scenario_dir = os.path.join(args.root_path, 'motion_forecasting', split)
        argoverse_scenario_dir = Path(argoverse_scenario_dir)
        all_scenario_files = sorted(argoverse_scenario_dir.rglob("*.parquet"))
        all_scenario_files = all_scenario_files[600:620]
        # ----------------------------------------------------------------------- #
        src_actor_trajectory_by_id: Dict[str, npt.NDArray] = {}
        tgt_actor_trajectory_by_id: Dict[str, npt.NDArray] = {}
        src_actor_offset_traj_id: Dict[str, npt.NDArray] = {}
        tgt_actor_offset_traj_id: Dict[str, npt.NDArray] = {}
        all_scene_data = []
        all_traj_data = []
        all_map_data = []
        # ----------------------------------------------------------------------- #
        for scenario_path in rich.progress.track(all_scenario_files, 'Processing data ...'):
            scenario_id = scenario_path.stem.split("_")[-1]
            static_map_path = scenario_path.parents[0] / f"log_map_archive_{scenario_id}.json"
            try:
                scenario = scenario_serialization.load_argoverse_scenario_parquet(scenario_path)
            except:
                print(Fore.RED + 'Fail to read: ' + Fore.RESET, scenario_path)
                return
            static_map = ArgoverseStaticMap.from_json(static_map_path)
            # ----------------------------------------------------------------------- #
            raw_scene_src_actor_traj_id: Dict[str, npt.NDArray] = {}
            raw_scene_tgt_actor_traj_id: Dict[str, npt.NDArray] = {}
            
            # Variables to store
            scene_src_actor_traj_id: Dict[str, npt.NDArray] = {}
            scene_tgt_actor_traj_id: Dict[str, npt.NDArray] = {}
            scene_src_actor_offTraj_id: Dict[str, npt.NDArray] = {}
            scene_tgt_actor_offTraj_id: Dict[str, npt.NDArray] = {}
            scene_data = {}
            # ----------------------------------------------------------------------- #
            # Get trajectories
            for track in scenario.tracks:
                # Only get the vehicles and dont save the ego-vehicle (AV), and 
                if track.object_type != ObjectType.VEHICLE or track.track_id == "AV":
                    continue
                # Only get the 'FOCAL TACK' and 'SCORED CARS'
                # if track.category != data_schema.TrackCategory.FOCAL_TRACK and track.category != data_schema.TrackCategory.SCORED_TRACK:
                #     continue
                if track.category != data_schema.TrackCategory.FOCAL_TRACK:
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
                
                raw_scene_src_actor_traj_id[track.track_id] = src_actor_trajectory
                raw_scene_tgt_actor_traj_id[track.track_id] = tgt_actor_trajectory
            # ----------------------------------------------------------------------- #
            if scenario.focal_track_id in raw_scene_tgt_actor_traj_id.keys():
                # Get the final observed trajectory of the focal agent
                focal_coordinate = raw_scene_src_actor_traj_id[scenario.focal_track_id][-1, 0:2]
                
                src_full_traj = raw_scene_src_actor_traj_id[scenario.focal_track_id][:, 0:2]
                tgt_full_traj = raw_scene_tgt_actor_traj_id[scenario.focal_track_id][:, 0:2]
                
                heading_vector = raw_scene_src_actor_traj_id[scenario.focal_track_id][-1, 2:4]
                sin_heading = heading_vector[0]
                cos_heading = heading_vector[1]
                # Get the focal heading
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
                
                
                # Transform the lane polylines
                scene_lanes_data: List[Dict] = []
                for id, lane_segment in static_map.vector_lane_segments.items():
                    left_lane_boundary = lane_segment.left_lane_boundary.xyz
                    right_lane_boundary = lane_segment.right_lane_boundary.xyz
                    
                    centerline = static_map.get_lane_segment_centerline(id)
                    
                    left_lane_boundary = np.append(left_lane_boundary, np.ones((left_lane_boundary.shape[0], 1)), axis=1)
                    right_lane_boundary = np.append(right_lane_boundary, np.ones((right_lane_boundary.shape[0], 1)), axis=1)
                    centerline = np.append(centerline, np.ones((centerline.shape[0], 1)), axis=1)
                    # Substract the center
                    left_lane_boundary = left_lane_boundary - np.append (focal_coordinate, [0, 0])
                    right_lane_boundary = right_lane_boundary - np.append (focal_coordinate, [0, 0])
                    centerline = centerline - np.append (focal_coordinate, [0, 0])
                    # Rotate
                    left_lane_boundary = np.dot(rot_matrix, left_lane_boundary.T).T
                    right_lane_boundary = np.dot(rot_matrix, right_lane_boundary.T).T
                    centerline = np.dot(rot_matrix, centerline.T).T
                    # Interpolate data to get all lines with the same size
                    left_lane_boundary = interp_arc (NUM_CENTERLINE_INTERP_PTS, points=left_lane_boundary[:, :3])
                    right_lane_boundary = interp_arc (NUM_CENTERLINE_INTERP_PTS, points=right_lane_boundary[:, :3])
                    
                    # save the data
                    lane_data = {"ID": lane_segment.id,
                                "left_lane_boundary": left_lane_boundary,
                                "right_lane_boundary": right_lane_boundary,
                                "centerline": centerline[:,:3],
                                "is_intersection": lane_segment.is_intersection,
                                "lane_type": lane_segment.lane_type,
                                "left_mark_type": LANE_MARKTYPE_DICT[lane_segment.left_mark_type],
                                "right_mark_type": LANE_MARKTYPE_DICT[lane_segment.right_mark_type],
                                # "right_neighbor_id": lane_segment.right_neighbor_id,
                                # "left_neighbor_id": lane_segment.left_neighbor_id
                                }
                    scene_lanes_data.append(lane_data)
                
                scene_agents_data: List[Dict] = []
                # Transform all trajectories
                for track_id in raw_scene_tgt_actor_traj_id.keys():
                    src_agent_coordinate = raw_scene_src_actor_traj_id[track_id][:, 0:2]
                    tgt_agent_coordinate = raw_scene_tgt_actor_traj_id[track_id][:, 0:2]
                    
                    src_agent_heading = np.arctan2(raw_scene_src_actor_traj_id[track_id][:,2], raw_scene_src_actor_traj_id[track_id][:,3])
                    tgt_agent_heading = np.arctan2(raw_scene_tgt_actor_traj_id[track_id][:,2], raw_scene_tgt_actor_traj_id[track_id][:,3])
                    
                    src_agent_velocity = raw_scene_src_actor_traj_id[track_id][:, 4:]
                    tgt_agent_velocity = raw_scene_tgt_actor_traj_id[track_id][:, 4:]
                    
                    # Add Z --> 0 and make matrix 4x4
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
                    
                    # Compute the offsets between points
                    src_actor_offset = np.vstack((src_agent_coordinate_tf[0], src_agent_coordinate_tf[1:] - src_agent_coordinate_tf[:-1]))
                    tgt_actor_offset = np.vstack((tgt_agent_coordinate_tf[0], tgt_agent_coordinate_tf[1:] - tgt_agent_coordinate_tf[:-1]))
                    
                    # Save the scene trajectory
                    scene_src_actor_traj_id[track_id] = src_agent_coordinate_tf
                    scene_tgt_actor_traj_id[track_id] = tgt_agent_coordinate_tf
                    scene_src_actor_offTraj_id[track_id] = src_actor_offset
                    scene_tgt_actor_offTraj_id[track_id] = tgt_actor_offset
                    # Save the trajectories by ID
                    src_actor_trajectory_by_id[track_id] = src_agent_coordinate_tf
                    tgt_actor_trajectory_by_id[track_id] = tgt_agent_coordinate_tf
                    src_actor_offset_traj_id[track_id] = src_actor_offset
                    tgt_actor_offset_traj_id[track_id] = tgt_actor_offset
                    # Save the data
                    agent_data = { "ID": track_id,
                                "historic": src_agent_coordinate_tf,
                                "future": tgt_agent_coordinate_tf,
                                "offset_historic": src_actor_offset,
                                "offset_future": tgt_actor_offset
                                }
                    scene_agents_data.append (agent_data)
                # ----------------------------------------------------------------------- #
                scene_data['agents'] = scene_agents_data
                scene_data['map'] = scene_lanes_data
                all_scene_data.append(scene_data)
                all_traj_data.append(scene_agents_data)
                all_map_data.append(scene_lanes_data)
        return all_traj_data, all_map_data
    # ---------------------------------------------------------------------------------------------------- #
    def __getitem__ (self, idx):
        sample = {}
        scene_traj_data = self.all_traj_data[idx]
        historic_trajectories = []
        future_trajectories = []
        historic_offset_trajectories = []
        future_offset_trajectories = []

        for traj_data in scene_traj_data:
            historic_trajectories.append (traj_data['historic'])
            future_trajectories.append(traj_data['future'])
            historic_offset_trajectories.append (traj_data['offset_historic'])
            future_offset_trajectories.append(traj_data['offset_future'])
        
        historic_trajectories = np.asarray(historic_trajectories)
        future_trajectories = np.asarray(future_trajectories)
        historic_offset_trajectories = np.asarray(historic_offset_trajectories)
        future_offset_trajectories = np.asarray(future_offset_trajectories)
        
        sample['historic'] = torch.tensor(historic_trajectories, dtype=torch.float32)
        sample['future'] = torch.tensor(future_trajectories, dtype=torch.float32)
        sample['offset_historic'] = torch.tensor(historic_offset_trajectories, dtype=torch.float32)
        sample['offset_future'] = torch.tensor(future_offset_trajectories, dtype=torch.float32)
        
        lanes = []
        scene_lane_data = self.all_map_data[idx]
        for lane_data in scene_lane_data:
            shape_vector = lane_data['left_lane_boundary'].shape
            is_intersection_v = np.repeat (int(lane_data['is_intersection']), shape_vector[0]).reshape(shape_vector[0], 1)
            left_mark_type_v = np.repeat (lane_data['left_mark_type'], shape_vector[0]).reshape(shape_vector[0], 1)
            right_mark_type_v = np.repeat (lane_data['right_mark_type'], shape_vector[0]).reshape(shape_vector[0], 1)
            id_v = np.repeat (lane_data['ID'], shape_vector[0]).reshape(shape_vector[0], 1)
            left_lane_feat = np.hstack ((lane_data['left_lane_boundary'], is_intersection_v, left_mark_type_v, id_v))
            right_lane_feat = np.hstack ((lane_data['right_lane_boundary'], is_intersection_v, right_mark_type_v, id_v))
            lanes.append(left_lane_feat)
            lanes.append(right_lane_feat)
        lanes = np.asarray(lanes)
        sample['lanes'] = torch.tensor(lanes, dtype=torch.float32).unsqueeze(0)
        return sample
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
    def predict(self):
        # Set network in eval mode
        self.model.eval()
        for idx, data in enumerate (self):
            with torch.no_grad():
                # Get the data
                historic_traj: torch.Tensor = data['historic'] # (bs, sequence length, feature number)
                future_traj: torch.Tensor = data['future']
                offset_future_traj: torch.Tensor = data['offset_future']
                lanes: torch.Tensor = data['lanes']
                
                # Pass to device
                historic_traj = historic_traj.to(self.device) 
                future_traj = future_traj.to(self.device)
                offset_future_traj = offset_future_traj.to(self.device)
                lanes = lanes.to(self.device)
                
                # Generate a square mask for the sequence
                # src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = self.create_mask(historic_traj, future_traj)
                # Output model
                                # x-7 ... x0 | x1 ... x7
                pred, conf = self.model (historic_traj, future_traj, lanes, src_mask=None, tgt_mask=None, src_padding_mask=None, tgt_padding_mask=None)
                
                plt.plot (historic_traj[0,:,0].cpu().numpy(), historic_traj[0,:,1].cpu().numpy(), '--o', color=(0,0,1), label='historical')
                for k in range(self.num_queries):
                    color = (1,0,0)
                    plt.plot (pred[0,k,:,0].cpu().numpy(), pred[0,k,:,1].cpu().numpy(), '--o', color=color, label='prediction')
                plt.plot (future_traj[0,:,0].cpu().numpy(), future_traj[0,:,1].cpu().numpy(), '--o', color=(0,1,0), label='GT')
                
                
                for lane in lanes[0]:
                    lane_cpu = lane.cpu().numpy()
                    plt.plot(lane_cpu[:, 0], lane_cpu[:, 1], "-", linewidth=1.0, color=(0,0,0), alpha=1.0) 
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
    cfg = Config.fromfile(args.path_2_configuration)
    
    # Model in prediction mode
    model_predict = TransformerPrediction (cfg)
    # Train the model
    model_predict.predict()
    # model_predict.visualization_prediction()
# ===================================================================================== #
if __name__ == '__main__':
    if not torch.cuda.is_available():
        print (Fore.CYAN + "Status: " + Fore.RED + "cuda not available" + Fore.RESET)
        exit(-1)
    main()