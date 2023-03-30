import torch
from torch.utils.data import Dataset
import os
from typing import Final, List, Optional, Sequence, Set, Tuple, Dict
from colorama import Fore
from pathlib import Path
from av2.datasets.motion_forecasting import scenario_serialization, data_schema
from av2.datasets.motion_forecasting.data_schema import ArgoverseScenario, ObjectType
from av2.map.map_api import ArgoverseStaticMap
from av2.utils.typing import NDArrayFloat, NDArrayInt
from rich.progress import track
import numpy as np
import numpy.typing as npt
import pickle
import threading

# ===================================================================================== #
# Configure constants
_OBS_DURATION_TIMESTEPS: Final[int] = 50 # The first 5 s of each scenario is denoted as the observed window
_PRED_DURATION_TIMESTEPS: Final[int] = 60 # The subsequent 6 s is denoted as the forecasted/predicted horizon.
_TOTAL_DURATION_TIMESTEPS: Final[int] = 110
_STATIC_OBJECT_TYPES: Set[ObjectType] = {
    ObjectType.STATIC,
    ObjectType.BACKGROUND,
    ObjectType.CONSTRUCTION,
    ObjectType.RIDERLESS_BICYCLE,
}
# ===================================================================================== #
class Av2MotionForecastingDataset (Dataset):
    """ PyTorch Dataset """
    def __init__(self, dataset_dir: str, split: str = "train", output_traj_size: int = 8, 
                 name_pickle: str = 'scored'):
        # Check if exists processed trajectories 
        self.path_2_traj = os.path.join('pickle_data/', split + '_trajectory_data_' + name_pickle + '.pickle')
        self.path_2_map = os.path.join('pickle_data/', split + '_map_data_' + name_pickle + '.pickle')
        
        # Read trajectory data
        if os.path.exists(self.path_2_traj):
            with open(self.path_2_traj, 'rb') as f:
                sample = pickle.load(f)
                self.all_traj_data = sample
        else:
            print (Fore.CYAN + 'Status: ' + Fore.RED + 'error to open trajectory pickle' + Fore.RED)
            exit(-1)
        # Read map data
        if os.path.exists(self.path_2_map):
            with open(self.path_2_map, 'rb') as f:
                sample = pickle.load(f)
                self.all_map_data = sample
        else:
            print (Fore.CYAN + 'Status: ' + Fore.RED + 'error to open map pickle' + Fore.RED)
            exit(-1)
        # ----------------------------------------------------------------------- #
        
    # ===================================================================================== #
    def __len__(self):
        return len (self.all_traj_data)
    # ===================================================================================== #
    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        """Get data

        Args:
            idx (_type_): _description_

        Returns:
            dict: Sequences for the encoder and for the decoder
        """
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
        sample['lanes'] = torch.tensor(lanes, dtype=torch.float32)
        return sample