import argparse
from pathlib import Path
import os
import pickle
from typing import Final, List, Optional, Sequence, Set, Tuple, Dict
import numpy as np
import numpy.typing as npt
from colorama import Fore
import rich.progress
import rich
import matplotlib.pyplot as plt
from av2.datasets.motion_forecasting import scenario_serialization, data_schema
from av2.datasets.motion_forecasting.data_schema import ArgoverseScenario, ObjectType
from av2.map.map_api import ArgoverseStaticMap
from av2.utils.typing import NDArrayFloat, NDArrayInt
from av2.geometry.interpolate import interp_arc
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
    default='target_simplified',
    help='Filename of the data')

args = parser.parse_args()
# ===================================================================================== #
# Configure constants
_OBS_DURATION_TIMESTEPS: Final[int] = 50 # The first 5 s of each scenario is denoted as the observed window
_PRED_DURATION_TIMESTEPS: Final[int] = 60 # The subsequent 6 s is denoted as the forecasted/predicted horizon.
_TOTAL_DURATION_TIMESTEPS: Final[int] = 110
NUM_CENTERLINE_INTERP_PTS: Final[int] = 10 # Polyline size
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
# ===================================================================================== #
def prepare_data_av2(split: str):
    argoverse_scenario_dir = os.path.join(args.root_path, 'motion_forecasting', split)
    argoverse_scenario_dir = Path(argoverse_scenario_dir)
    all_scenario_files = sorted(argoverse_scenario_dir.rglob("*.parquet"))
    # ----------------------------------------------------------------------- #
    all_scene_data = []
    # ----------------------------------------------------------------------- #
    for scenario_path in rich.progress.track(all_scenario_files, 'Processing data ...'):
        scenario_id = scenario_path.stem.split("_")[-1]
        static_map_path = scenario_path.parents[0] / f"log_map_archive_{scenario_id}.json"
        try:
            scenario = scenario_serialization.load_argoverse_scenario_parquet(scenario_path)
        except:
            print(Fore.RED + 'Fail to read: ' + Fore.RESET, scenario_path)
            return
        # static_map = ArgoverseStaticMap.from_json(static_map_path)
        # ----------------------------------------------------------------------- #
        raw_scene_src_actor_traj_id: Dict[str, npt.NDArray] = {}
        raw_scene_tgt_actor_traj_id: Dict[str, npt.NDArray] = {}
        # Variables to store
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
                
                # Save the data
                agent_data = { "ID": track_id,
                               "historic": src_agent_coordinate_tf,
                               "future": tgt_agent_coordinate_tf,
                               "offset_historic": src_actor_offset,
                               "offset_future": tgt_actor_offset
                             }
                scene_agents_data.append (agent_data)
            # Save the focal agent data
            focal_agent_data = {}
            focal_agent_data['focal_coordinate'] = focal_coordinate
            focal_agent_data['focal_rot_matrix'] = rot_matrix
            # ----------------------------------------------------------------------- #
            scene_data['agents'] = scene_agents_data
            scene_data['focal_agent'] = focal_agent_data
            scene_data['map_path'] = static_map_path
            all_scene_data.append(scene_data)
            # ----------------------------------------------------------------------- #
            # Plot
            # for lane_data in scene_lanes_data:
            #     polylines = [lane_data['left_lane_boundary'], lane_data['right_lane_boundary']]
            #     centerlines = lane_data['centerline']
                
            #     for polyline in polylines:
            #         plt.plot(polyline[:, 0], polyline[:, 1], "-", linewidth=1.0, color=(0,0,0), alpha=1.0)
            #     plt.plot(centerlines[:, 0], centerlines[:, 1], "-", linewidth=1.0, color='y', alpha=1.0)
            # for track_id in scene_src_actor_traj_id.keys():
            #     src_traj = scene_src_actor_traj_id[track_id]
            #     tgt_traj = scene_tgt_actor_traj_id[track_id]
            #     plt.plot(src_traj[:, 0], src_traj[:, 1], "-", linewidth=1.0, color='b', alpha=1.0)
            #     plt.plot(tgt_traj[:, 0], tgt_traj[:, 1], "-", linewidth=1.0, color='g', alpha=1.0)
            # plt.show()
        else:
            # Not found focal agent or target agent
            # Delete scenario
            for track in scenario.tracks:
                if track.track_id in raw_scene_tgt_actor_traj_id.keys():
                    del raw_scene_src_actor_traj_id[track.track_id]
                    del raw_scene_tgt_actor_traj_id[track.track_id]
    # ----------------------------------------------------------------------- #
    # Print info
    print (Fore.CYAN + 'Size scene data: ' + Fore.WHITE + str(len(all_scene_data)) + Fore.RESET)
    # ----------------------------------------------------------------------- #
    # Save the data in a pickle
    path_2_save_scenes = os.path.join('pickle_data/', split + '_scenes_data_' + args.output_filename + '.pickle')
    with open(path_2_save_scenes, 'wb') as f:
        pickle.dump(all_scene_data, f, pickle.HIGHEST_PROTOCOL)
    print (Fore.CYAN + 'Data saved in: ' + Fore.WHITE + path_2_save_scenes + Fore.RESET)
    
# ===================================================================================== #
if __name__ == '__main__':
    if args.dataset == 'av2':
        splits = ['train', 'val']
        for split in splits:
            prepare_data_av2(split)
