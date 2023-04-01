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
from av2.geometry.interpolate import interp_arc
import matplotlib.pyplot as plt
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
NUM_CENTERLINE_INTERP_PTS: Final[int] = 10 # Polyline size
# ===================================================================================== #
class Av2MotionForecastingDataset (Dataset):
    """ PyTorch Dataset """
    def __init__(self, dataset_dir: str, split: str = "train", output_traj_size: int = 8, 
                 name_pickle: str = 'scored'):
        # Check if exists processed trajectories 
        self.path_2_scenes_data = os.path.join('pickle_data/', split + '_scenes_data_' + name_pickle + '.pickle')
        
        # Read scenes data
        if os.path.exists(self.path_2_scenes_data):
            with open(self.path_2_scenes_data, 'rb') as f:
                sample = pickle.load(f)
                self.scenes_data = sample
        else:
            raise FileNotFoundError('Error to open scenes data')
        # ----------------------------------------------------------------------- #
        
    # ===================================================================================== #
    def __len__(self):
        return len (self.scenes_data)
    # ===================================================================================== #
    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        """Get data

        Args:
            idx (_type_): _description_

        Returns:
            dict: Sequences for the encoder and for the decoder
        """
        sample = {}
        scene_traj_data = self.scenes_data[idx]['agents']
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
        # Get focal agent data 
        focal_agent_data = self.scenes_data[idx]['focal_agent']
        focal_coordinate = focal_agent_data['focal_coordinate']
        rot_matrix = focal_agent_data['focal_rot_matrix']
        
        # Process map data
        static_map_path = self.scenes_data[idx]['map_path']
        static_map = ArgoverseStaticMap.from_json(static_map_path)
        scene_lanes_data: List[Dict] = []
        
        lanes = []
        for id, lane_segment in static_map.vector_lane_segments.items():
            left_lane_boundary = lane_segment.left_lane_boundary.xyz
            right_lane_boundary = lane_segment.right_lane_boundary.xyz
            
            # centerline = static_map.get_lane_segment_centerline(id)
            
            left_lane_boundary = np.append(left_lane_boundary, np.ones((left_lane_boundary.shape[0], 1)), axis=1)
            right_lane_boundary = np.append(right_lane_boundary, np.ones((right_lane_boundary.shape[0], 1)), axis=1)
            # centerline = np.append(centerline, np.ones((centerline.shape[0], 1)), axis=1)
            # Substract the center
            left_lane_boundary = left_lane_boundary - np.append (focal_coordinate, [0, 0])
            right_lane_boundary = right_lane_boundary - np.append (focal_coordinate, [0, 0])
            # centerline = centerline - np.append (focal_coordinate, [0, 0])
            # Rotate
            left_lane_boundary = np.dot(rot_matrix, left_lane_boundary.T).T
            right_lane_boundary = np.dot(rot_matrix, right_lane_boundary.T).T
            # centerline = np.dot(rot_matrix, centerline.T).T
            # Interpolate data to get all lines with the same size
            left_lane_boundary = interp_arc (NUM_CENTERLINE_INTERP_PTS, points=left_lane_boundary[:, :3])
            right_lane_boundary = interp_arc (NUM_CENTERLINE_INTERP_PTS, points=right_lane_boundary[:, :3])
            
            shape_vector = left_lane_boundary.shape
            is_intersection_v = np.repeat (int(lane_segment.is_intersection), shape_vector[0]).reshape(shape_vector[0], 1)
            
            left_mark_type_v = np.repeat (LANE_MARKTYPE_DICT[lane_segment.left_mark_type], shape_vector[0]).reshape(shape_vector[0], 1)
            right_mark_type_v = np.repeat (LANE_MARKTYPE_DICT[lane_segment.right_mark_type], shape_vector[0]).reshape(shape_vector[0], 1)
            
            id_v = np.repeat (lane_segment.id, shape_vector[0]).reshape(shape_vector[0], 1)
            
            left_lane_feat = np.hstack ((left_lane_boundary, is_intersection_v, left_mark_type_v, id_v))
            right_lane_feat = np.hstack ((right_lane_boundary, is_intersection_v, right_mark_type_v, id_v))
            
            lanes.append(left_lane_feat)
            lanes.append(right_lane_feat)
        sample['lanes'] = torch.tensor(lanes, dtype=torch.float32)
        return sample