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
_OBS_DURATION_TIMESTEPS: Final[int] = 8 # The first 5 s of each scenario is denoted as the observed window
_PRED_DURATION_TIMESTEPS: Final[int] = 12 # The subsequent 6 s is denoted as the forecasted/predicted horizon.
_TOTAL_DURATION_TIMESTEPS: Final[int] = 20
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
                 filename_pickle_src: str = 'src_trajectory_data', filename_pickle_tgt: str = 'tgt_trajectory_data', load_pickle = True, save_pickle=False):
        
        self.src_actor_trajectory_by_id: Dict[str, npt.NDArray] = {}
        self.tgt_actor_trajectory_by_id: Dict[str, npt.NDArray] = {}
        # Check if exists processed trajectories 
        self.path_2_save_src = os.path.join('pickle_data/', str(split + '_' + filename_pickle_src))
        self.path_2_save_tgt = os.path.join('pickle_data/', str(split + '_' + filename_pickle_tgt))
        if load_pickle:
            with open(self.path_2_save_src, 'rb') as f:
                self.src_actor_trajectory_by_id = pickle.load(f)
            with open(self.path_2_save_tgt, 'rb') as f:
                self.tgt_actor_trajectory_by_id = pickle.load(f)
        else:
            argoverse_scenario_dir = os.path.join(dataset_dir, 'argoverse2', 'motion_forecasting', split)
            argoverse_scenario_dir = Path(argoverse_scenario_dir)
            all_scenario_files = sorted(argoverse_scenario_dir.rglob("*.parquet"))
            
            begin = 0
            end = 0
            threads = list()
            scenario_path_splitted = list()
            num_threads = 32
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

        # Save the data if is not processed
        if save_pickle:
            self.__save_trajectories()
        self.src_sequences, self.tgt_sequences = self.__prepare_data()
    # ===================================================================================== #   
    def __save_trajectories (self):
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
        try:
            scenario = scenario_serialization.load_argoverse_scenario_parquet(scenario_path)
        except:
            print(Fore.RED + 'Fail to read: ' + Fore.RESET, scenario_path)
            return
        # static_map = ArgoverseStaticMap.from_json(static_map_path)
        
        scenario_src_actor_trajectory_by_id: Dict[str, npt.NDArray] = {}
        scenario_tgt_actor_trajectory_by_id: Dict[str, npt.NDArray] = {}
        
        # Get trajectories
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
        else:
            # Not found focal agent or target agent
            # Delete scenario
            for track in scenario.tracks:
                if track.track_id in scenario_tgt_actor_trajectory_by_id.keys():
                    del scenario_src_actor_trajectory_by_id[track.track_id]
                    del scenario_tgt_actor_trajectory_by_id[track.track_id]
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