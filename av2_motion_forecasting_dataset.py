import torch
from torch.utils.data import Dataset
import os
from typing import Final, List, Optional, Sequence, Set, Tuple, Dict

from pathlib import Path
from av2.datasets.motion_forecasting import scenario_serialization
from av2.datasets.motion_forecasting.data_schema import ArgoverseScenario, ObjectType
from av2.map.map_api import ArgoverseStaticMap
from av2.utils.typing import NDArrayFloat, NDArrayInt
from rich.progress import track
import numpy as np
import numpy.typing as npt

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
    def __init__(self, dataset_dir: str, split: str = "train", sequence_length: int = 8):
        argoverse_scenario_dir = os.path.join(dataset_dir, 'argoverse2', 'motion_forecasting', split)
        argoverse_scenario_dir = Path(argoverse_scenario_dir)
        all_scenario_files = sorted(argoverse_scenario_dir.rglob("*.parquet"))
        
        self.src_actor_trajectory_by_id: Dict[str, npt.NDArray] = {}
        self.tgt_actor_trajectory_by_id: Dict[str, npt.NDArray] = {}
        # Check if exists processed trajectories 
        self.path_2_save_src = os.path.join('pickle_data/', str(split + '_src_trajectory_data.pickle'))
        self.path_2_save_tgt = os.path.join('pickle_data/', str(split + '_tgt_trajectory_data.pickle'))
        if load_pickle:
            with open(self.path_2_save_src, 'rb') as f:
                self.src_actor_trajectory_by_id = pickle.load(f)
            with open(self.path_2_save_tgt, 'rb') as f:
                self.tgt_actor_trajectory_by_id = pickle.load(f)
        else:
            """
            idx = 0
        for scenario_path in track(all_scenario_files):
            self.__generate_scenario(scenario_path)
            idx += 1
            if idx > 5:
                break
        self.src_sequences, self.tgt_sequences = self.__prepare_data()
    # ===================================================================================== #   
    def __save_trajectories (self):
        print ('saving')
        # Save the trajectories in a pickle
        with open(self.path_2_save_src, 'wb') as f:
            pickle.dump(self.src_actor_trajectory_by_id, f)
        with open(self.path_2_save_tgt, 'wb') as f:
            pickle.dump(self.tgt_actor_trajectory_by_id, f)
    # ===================================================================================== #
    def __generate_scenario_parallel (self, scenarios_path: List) -> None:
        for scenario_path in scenarios_path:
            self.__generate_scenario(scenario_path)
    # ===================================================================================== #
    def __generate_scenario (self, scenario_path: Path) -> None:
        """_summary_

        Args:
            scenario_path (Path): Path to the parquet file corresponding to the Argoverse scenario
        """
        # ----------------------------------------------------------------------- #
        scenario_id = scenario_path.stem.split("_")[-1]
        static_map_path = scenario_path.parents[0] / f"log_map_archive_{scenario_id}.json"
        scenario = scenario_serialization.load_argoverse_scenario_parquet(scenario_path)
        static_map = ArgoverseStaticMap.from_json(static_map_path)
        
        # Get trajectories
        for track in scenario.tracks:
            # Only get the vehicles
            if track.object_type != ObjectType.VEHICLE:
                continue
            # Get timesteps for which actor data is valid
            actor_timesteps: NDArrayInt = np.array( [object_state.timestep for object_state in track.object_states if object_state.timestep < _TOTAL_DURATION_TIMESTEPS] )
            if actor_timesteps.shape[0] < 1 or actor_timesteps.shape[0] != _TOTAL_DURATION_TIMESTEPS:
                continue
            # Get actor trajectory and heading history and instantaneous velocity
            actor_trajectory: NDArrayFloat = np.array( [list(object_state.position) + [np.sin(object_state.heading), np.cos(object_state.heading)] + list(object_state.velocity) for object_state in track.object_states if object_state.timestep < _TOTAL_DURATION_TIMESTEPS] )
            # Get source actor trajectory and heading history -> observerd or historical trajectory
            src_actor_trajectory = actor_trajectory[:_OBS_DURATION_TIMESTEPS]
            # Get target actor trajectory and heading history -> forescated or predicted trajectory
            tgt_actor_trajectory = actor_trajectory[_OBS_DURATION_TIMESTEPS:_TOTAL_DURATION_TIMESTEPS]
            
            self.src_actor_trajectory_by_id[track.track_id] = src_actor_trajectory
            self.tgt_actor_trajectory_by_id[track.track_id] = tgt_actor_trajectory
    # ===================================================================================== #
    def __prepare_data (self) -> list:
        src_sequences = []
        tgt_sequences = []
        for key in self.src_actor_trajectory_by_id.keys():
            src_sequences.append(self.src_actor_trajectory_by_id[key].tolist())
            tgt_sequences.append(self.tgt_actor_trajectory_by_id[key].tolist())
        return np.array(src_sequences), np.array(tgt_sequences)
    # ===================================================================================== #
    def __len__(self):
        return len (self.src_sequences)
    # ===================================================================================== #
    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        """Get data

        Args:
            idx (_type_): _description_

        Returns:
            dict: Sequences for the encoder and for the decoder
        """
        sample = {}
        sample ['src'] = torch.tensor(self.src_sequences[idx], dtype=torch.float32)
        sample ['tgt'] = torch.tensor(self.tgt_sequences[idx], dtype=torch.float32)
        return sample
        
            
                
                