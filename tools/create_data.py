import argparse
from pathlib import Path
import os
import pickle
from typing import Final, List, Optional, Sequence, Set, Tuple, Dict
import numpy as np
import numpy.typing as npt
from colorama import Fore
import matplotlib.pyplot as plt
from av2.datasets.motion_forecasting import scenario_serialization, data_schema
from av2.datasets.motion_forecasting.data_schema import ArgoverseScenario, ObjectType
from av2.map.map_api import ArgoverseStaticMap
from av2.utils.typing import NDArrayFloat, NDArrayInt
# ===================================================================================== #
parser = argparse.ArgumentParser(description='Data converter arg parser')
parser.add_argument('dataset', metavar='av2', help='name of the dataset')
parser.add_argument(
    '--root-path',
    type=str,
    default='/datasets/argoverse2/',
    help='specify the root path of dataset')
parser.add_argument(
    '--output_filename',
    type=str,
    default='scored',
    help='Filename of the data')

args = parser.parse_args()
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
def prepare_data_av2(split: str):
    argoverse_scenario_dir = os.path.join(args.root_path, 'motion_forecasting', split)
    argoverse_scenario_dir = Path(argoverse_scenario_dir)
    all_scenario_files = sorted(argoverse_scenario_dir.rglob("*.parquet"))
    # ----------------------------------------------------------------------- #
    src_actor_trajectory_by_id: Dict[str, npt.NDArray] = {}
    tgt_actor_trajectory_by_id: Dict[str, npt.NDArray] = {}
    src_actor_offset_traj_id: Dict[str, npt.NDArray] = {}
    tgt_actor_offset_traj_id: Dict[str, npt.NDArray] = {}
    # ----------------------------------------------------------------------- #
    for scenario_path in all_scenario_files:
        scenario_id = scenario_path.stem.split("_")[-1]
        static_map_path = scenario_path.parents[0] / f"log_map_archive_{scenario_id}.json"
        try:
            scenario = scenario_serialization.load_argoverse_scenario_parquet(scenario_path)
        except:
            print(Fore.RED + 'Fail to read: ' + Fore.RESET, scenario_path)
            return
        static_map = ArgoverseStaticMap.from_json(static_map_path)
        scenario_src_actor_trajectory_by_id: Dict[str, npt.NDArray] = {}
        scenario_tgt_actor_trajectory_by_id: Dict[str, npt.NDArray] = {}
        
        polylines_id: Dict[int, List[npt.NDArray]] = {}
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
            
            scenario_src_actor_trajectory_by_id[track.track_id] = src_actor_trajectory
            scenario_tgt_actor_trajectory_by_id[track.track_id] = tgt_actor_trajectory
        # ----------------------------------------------------------------------- #
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
            
            # Transform the lane polylines
            for lane_segment in static_map.vector_lane_segments.values():
                
                left_lane_boundary = lane_segment.left_lane_boundary.xyz
                right_lane_boundary = lane_segment.right_lane_boundary.xyz
                
                left_lane_boundary = np.append(left_lane_boundary, np.ones((left_lane_boundary.shape[0], 1)), axis=1)
                right_lane_boundary = np.append(right_lane_boundary, np.ones((right_lane_boundary.shape[0], 1)), axis=1)
                # Substract the center
                left_lane_boundary = left_lane_boundary - np.append (focal_coordinate, [0, 0])
                right_lane_boundary = right_lane_boundary - np.append (focal_coordinate, [0, 0])
                
                left_lane_boundary = np.dot(rot_matrix, left_lane_boundary.T).T
                right_lane_boundary = np.dot(rot_matrix, right_lane_boundary.T).T
                
                polylines = [left_lane_boundary[:,0:3], right_lane_boundary[:,0:3]]
                polylines_id[lane_segment.id] = polylines
            
            # Transform all trajectories
            for track_id in scenario_tgt_actor_trajectory_by_id.keys():
                src_agent_coordinate = scenario_src_actor_trajectory_by_id[track_id][:, 0:2]
                tgt_agent_coordinate = scenario_tgt_actor_trajectory_by_id[track_id][:, 0:2]
                
                src_agent_heading = np.arctan2(scenario_src_actor_trajectory_by_id[track_id][:,2], scenario_src_actor_trajectory_by_id[track_id][:,3])
                tgt_agent_heading = np.arctan2(scenario_tgt_actor_trajectory_by_id[track_id][:,2], scenario_tgt_actor_trajectory_by_id[track_id][:,3])
                
                src_agent_velocity = scenario_src_actor_trajectory_by_id[track_id][:, 4:]
                tgt_agent_velocity = scenario_tgt_actor_trajectory_by_id[track_id][:, 4:]
                
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
                
                # Save the trajectory
                src_actor_trajectory_by_id[track_id] = src_agent_coordinate_tf
                tgt_actor_trajectory_by_id[track_id] = tgt_agent_coordinate_tf
                src_actor_offset_traj_id[track_id] = np.vstack((src_agent_coordinate_tf[0], src_agent_coordinate_tf[1:] - src_agent_coordinate_tf[:-1]))
                tgt_actor_offset_traj_id[track_id] = np.vstack((tgt_agent_coordinate_tf[0], tgt_agent_coordinate_tf[1:] - tgt_agent_coordinate_tf[:-1]))
            # ----------------------------------------------------------------------- #
            # Plot
            # for lane_id in polylines_id.keys():
            #     polylines = polylines_id[lane_id]
            #     for polyline in polylines:
            #         plt.plot(polyline[:, 0], polyline[:, 1], "-", linewidth=1.0, color='r', alpha=1.0)
            # for id_agent in src_actor_trajectory_by_id.keys():
            #     src_traj = src_actor_trajectory_by_id[id_agent]
            #     tgt_traj = tgt_actor_trajectory_by_id[id_agent]
            #     plt.plot(src_traj[:, 0], src_traj[:, 1], "-", linewidth=1.0, color='b', alpha=1.0)
            #     plt.plot(tgt_traj[:, 0], tgt_traj[:, 1], "-", linewidth=1.0, color='g', alpha=1.0)
            # plt.show()
                
        else:
            # Not found focal agent or target agent
            # Delete scenario
            for track in scenario.tracks:
                if track.track_id in scenario_tgt_actor_trajectory_by_id.keys():
                    del scenario_src_actor_trajectory_by_id[track.track_id]
                    del scenario_tgt_actor_trajectory_by_id[track.track_id]
    # ----------------------------------------------------------------------- #
    src_sequences = []
    tgt_sequences = []
    src_offset_sequences = []
    tgt_offset_sequences = []
    for key in src_actor_trajectory_by_id.keys():
        src_sequences.append(src_actor_trajectory_by_id[key].tolist())
        tgt_sequences.append(tgt_actor_trajectory_by_id[key].tolist())
        src_offset_sequences.append(src_actor_offset_traj_id[key].tolist())
        tgt_offset_sequences.append(tgt_actor_offset_traj_id[key].tolist())
    np_src_sequences = np.array(src_sequences)
    np_tgt_sequences = np.array(tgt_sequences)
    np_src_offset_sequences = np.array(src_offset_sequences)
    np_tgt_offset_sequences = np.array(tgt_offset_sequences)
    # ----------------------------------------------------------------------- #
    # Save the trajectories in a pickle
    sample = {}
    sample['src'] = src_actor_trajectory_by_id
    sample['tgt'] = tgt_actor_trajectory_by_id
    sample['offset_src'] = src_actor_offset_traj_id
    sample['offset_tgt'] = tgt_actor_offset_traj_id
    
    path_2_save = os.path.join('pickle_data/', split + '_trajectory_data_' + args.output_filename + '.pickle')
    with open(path_2_save, 'wb') as f:
        pickle.dump(sample, f)
# ===================================================================================== #
if __name__ == '__main__':
    if args.dataset == 'av2':
        splits = ['train', 'val']
        for split in splits:
            prepare_data_av2(split)
