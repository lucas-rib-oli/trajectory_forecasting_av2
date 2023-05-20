import argparse
import io
import rich
import os
import cv2
from pathlib import Path
from av2.datasets.motion_forecasting import scenario_serialization, data_schema
from av2.datasets.motion_forecasting.data_schema import ObjectType, ArgoverseScenario, TrackCategory, ObjectState
from av2.map.map_api import ArgoverseStaticMap
from av2.geometry.interpolate import interp_arc
from typing import Dict, List, Final, Set, Tuple, Optional, Sequence
from colorama import Fore
import numpy as np
from av2.utils.typing import NDArrayFloat, NDArrayInt
import numpy.typing as npt
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image as img
from PIL.Image import Image

# ===================================================================================== #
_PlotBounds = Tuple[float, float, float, float]

# Configure constants
_OBS_DURATION_TIMESTEPS: Final[int] = 50 # The first 5 s of each scenario is denoted as the observed window
_PRED_DURATION_TIMESTEPS: Final[int] = 60 # The subsequent 6 s is denoted as the forecasted/predicted horizon.
_TOTAL_DURATION_TIMESTEPS: Final[int] = 110
NUM_CENTERLINE_INTERP_PTS: Final[int] = 10 # Polyline size

_ESTIMATED_VEHICLE_LENGTH_M: Final[float] = 4.0
_ESTIMATED_VEHICLE_WIDTH_M: Final[float] = 2.0
_ESTIMATED_CYCLIST_LENGTH_M: Final[float] = 2.0
_ESTIMATED_CYCLIST_WIDTH_M: Final[float] = 0.7
_PLOT_BOUNDS_BUFFER_M: Final[float] = 30.0

_DRIVABLE_AREA_COLOR: Final[str] = "#7A7A7A"
_LANE_SEGMENT_COLOR: Final[str] = "#E0E0E0"

_DEFAULT_ACTOR_COLOR: Final[str] = "#D3E8EF"
_FOCAL_AGENT_COLOR: Final[str] = "#ECA25B"
_AV_COLOR: Final[str] = "#007672"
_BOUNDING_BOX_ZORDER: Final[int] = 100  # Ensure actor bounding boxes are plotted on top of all map elements

_STATIC_OBJECT_TYPES: Set[ObjectType] = {
    ObjectType.STATIC,
    ObjectType.BACKGROUND,
    ObjectType.CONSTRUCTION,
    ObjectType.RIDERLESS_BICYCLE,
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
parser.add_argument(
    '--split',
    type=str,
    default='val',
    help='specify the split of dataset')
args = parser.parse_args()
# ===================================================================================== #
def normalice_heading (angle):
    norm_angle = (angle + np.pi) % (2 * np.pi) - np.pi
    return norm_angle
# ===================================================================================== #
def get_av2_data (split: str, idx: int = 150) -> Tuple[ArgoverseScenario, ArgoverseStaticMap]:
    argoverse_scenario_dir = os.path.join(args.root_path, 'motion_forecasting', split)
    argoverse_scenario_dir = Path(argoverse_scenario_dir)
    all_scenario_files = sorted(argoverse_scenario_dir.rglob("*.parquet"))
    scenario_path = all_scenario_files[idx]
    # ----------------------------------------------------------------------- #
    scenario_id = scenario_path.stem.split("_")[-1]
    static_map_path = scenario_path.parents[0] / f"log_map_archive_{scenario_id}.json"
    try:
        scenario = scenario_serialization.load_argoverse_scenario_parquet(scenario_path)
    except:
        print(Fore.RED + 'Fail to read: ' + Fore.RESET, scenario_path)
        return
    static_map = ArgoverseStaticMap.from_json(static_map_path)
    
    return scenario, static_map

def visualize_scenario(scenario: ArgoverseScenario, static_map: ArgoverseStaticMap) -> None:
    """Build dynamic visualization for all tracks and the local map associated with an Argoverse scenario.

    Note: This function uses OpenCV to create a MP4 file using the MP4V codec.

    Args:
        scenario: Argoverse scenario to visualize.
        static_map: Local static map elements associated with `scenario`.
        save_path: Path where output MP4 video should be saved.
    """
    # Build each frame for the video
    frames: List[Image] = []
    plot_bounds: _PlotBounds = (0, 0, 0, 0)
    # Get focal/target agent
    for track in scenario.tracks:
        if track.category == data_schema.TrackCategory.FOCAL_TRACK:
            focal_track = track
    
    # Get the final observed trajectory of the focal agent
    last_obs_focal_state = focal_track.object_states[_OBS_DURATION_TIMESTEPS - 1]
    x0, y0 = last_obs_focal_state.position
    
    last_obs_focal_heading = last_obs_focal_state.heading
    last_obs_sin_heading = np.sin(-last_obs_focal_heading)
    last_obs_cos_heading = np.cos(-last_obs_focal_heading)
    
    
    rot_matrix = np.array([[last_obs_cos_heading, -last_obs_sin_heading, 0],
                           [last_obs_sin_heading,  last_obs_cos_heading, 0],
                           [                   0,                     0, 1]])
    tf_matrix = np.array([[last_obs_cos_heading, -last_obs_sin_heading, 0.0, -x0*np.cos(-last_obs_focal_heading) + y0*np.sin(-last_obs_focal_heading)],
                          [last_obs_sin_heading,  last_obs_cos_heading, 0.0, -x0*np.sin(-last_obs_focal_heading) - y0*np.cos(-last_obs_focal_heading)],
                          [                   0.0,                 0.0, 1.0,                                                                    0.0],
                          [                   0.0,                 0.0, 0.0,                                                                    1.0]])
    
    _, ax = plt.subplots()
    # ----------------------------------------------------------------------- #
    # Transform coordinates to target-centric
    for track in scenario.tracks:
        transformed_object_states: List[ObjectState] = []
        for object_state in track.object_states:
            transformed_object_state = object_state
            # Transform position
            x, y = object_state.position
            position = np.array([x, y, 0.0, 1.0])
            transformed_position = np.dot(tf_matrix, position)
            # Transform heading
            transformed_object_state.heading = normalice_heading(object_state.heading - last_obs_focal_heading)
            # Transform velocity
            vx, vy = object_state.velocity
            velocity = np.array([vx, vy, 0.0])
            transformed_object_state.velocity = np.dot(rot_matrix, velocity)
            # Set the transformed pos
            transformed_object_state.position = transformed_position[:2]
            # Save
            transformed_object_states.append(transformed_object_state)
        track.object_states = transformed_object_states
    _plot_actor_tracks(ax, scenario, _OBS_DURATION_TIMESTEPS-1)
    # ----------------------------------------------------------------------- #
    # Transform map to target-centric
    for lane_segment in static_map.vector_lane_segments.values():
        left_lane_boundary = lane_segment.left_lane_boundary.xyz
        right_lane_boundary = lane_segment.right_lane_boundary.xyz
                
        left_lane_boundary = np.append(left_lane_boundary, np.ones((left_lane_boundary.shape[0], 1)), axis=1)
        right_lane_boundary = np.append(right_lane_boundary, np.ones((right_lane_boundary.shape[0], 1)), axis=1)
        transformed_left_lane_boundary = np.dot(tf_matrix, left_lane_boundary.T).T[:, :3]
        transformed_right_lane_boundary = np.dot(tf_matrix, right_lane_boundary.T).T[:, :3]
        
        _plot_polylines(
            [
                transformed_left_lane_boundary,
                transformed_right_lane_boundary,
            ],
            line_width=0.5,
            color=_LANE_SEGMENT_COLOR,
        )
        
    # ----------------------------------------------------------------------- #
    # Transform drivable area to target-centric  
    for drivable_area in static_map.vector_drivable_areas.values():
        drivable_area_xyz = np.append(drivable_area.xyz, np.ones((drivable_area.xyz.shape[0], 1)), axis=1)
        transformed_drivable_area_xyz = np.dot(tf_matrix, drivable_area_xyz.T).T[:, :3]
        
        _plot_polygons([transformed_drivable_area_xyz], alpha=0.5, color=_DRIVABLE_AREA_COLOR)
    # ----------------------------------------------------------------------- #
    for ped_xing in static_map.vector_pedestrian_crossings.values():
        edge1_xyz = np.append(ped_xing.edge1.xyz, np.ones((ped_xing.edge1.xyz.shape[0], 1)), axis=1)
        edge2_xyz = np.append(ped_xing.edge2.xyz, np.ones((ped_xing.edge2.xyz.shape[0], 1)), axis=1)
        transformed_edge1_xyz = np.dot(tf_matrix, edge1_xyz.T).T[:, :3]
        transformed_edge2_xyz = np.dot(tf_matrix, edge2_xyz.T).T[:, :3]
        
        _plot_polylines([transformed_edge1_xyz, transformed_edge2_xyz ], alpha=1.0, color=_LANE_SEGMENT_COLOR)
        
    # ----------------------------------------------------------------------- #
    # Dibujar el vector del eje X
    arrow_x = ax.arrow(0, 0, 10, 0, head_width=3.0, head_length=3.0, fc='red', ec='red')
    ax.text(10.1, 2.5, 'X', color='red')

    # Dibujar el vector del eje Y
    arrow_y = ax.arrow(0, 0, 0, 10, head_width=3.0, head_length=3.0, fc='green', ec='green')
    ax.text(2.0, 10.1, 'Y', color='green')
    
    arrow_x.set_zorder(10)
    arrow_y.set_zorder(10)
    
    # Etiquetas de los ejes
    ax.set_xlabel('Eje X')
    ax.set_ylabel('Eje Y')
    plt.show()
    for timestep in range(_OBS_DURATION_TIMESTEPS + _PRED_DURATION_TIMESTEPS):
        _, ax = plt.subplots()

        # Plot static map elements and actor tracks
        # _plot_static_map_elements(static_map)
        # cur_plot_bounds = _plot_actor_tracks(ax, scenario, timestep)
        # if cur_plot_bounds:
        #     plot_bounds = cur_plot_bounds

        # Set map bounds to capture focal trajectory history (with fixed buffer in all directions)
        # plt.xlim(plot_bounds[0] - _PLOT_BOUNDS_BUFFER_M, plot_bounds[1] + _PLOT_BOUNDS_BUFFER_M)
        # plt.ylim(plot_bounds[2] - _PLOT_BOUNDS_BUFFER_M, plot_bounds[3] + _PLOT_BOUNDS_BUFFER_M)
        # plt.gca().set_aspect("equal", adjustable="box")

        # Minimize plot margins and make axes invisible
        plt.gca().set_axis_off()
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())

        # Save plotted frame to in-memory buffer
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        plt.close()
        buf.seek(0)
        frame = img.open(buf)
        cv2.imshow('frame: ', frame)
        cv2.waitKey(0)
        frames.append(frame)

    # Write buffered frames to MP4V-encoded video
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vid_path = 'prueba.mp4'
    video = cv2.VideoWriter(vid_path, fourcc, fps=10, frameSize=frames[0].size)
    for i in range(len(frames)):
        frame_temp = frames[i].copy()
        video.write(cv2.cvtColor(np.array(frame_temp), cv2.COLOR_RGB2BGR))
    video.release()


def _plot_static_map_elements(static_map: ArgoverseStaticMap, show_ped_xings: bool = False) -> None:
    """Plot all static map elements associated with an Argoverse scenario.

    Args:
        static_map: Static map containing elements to be plotted.
        show_ped_xings: Configures whether pedestrian crossings should be plotted.
    """
    # Plot drivable areas
    for drivable_area in static_map.vector_drivable_areas.values():
        _plot_polygons([drivable_area.xyz], alpha=0.5, color=_DRIVABLE_AREA_COLOR)

    # Plot lane segments
    for lane_segment in static_map.vector_lane_segments.values():
        _plot_polylines(
            [
                lane_segment.left_lane_boundary.xyz,
                lane_segment.right_lane_boundary.xyz,
            ],
            line_width=0.5,
            color=_LANE_SEGMENT_COLOR,
        )

    # Plot pedestrian crossings
    if show_ped_xings:
        for ped_xing in static_map.vector_pedestrian_crossings.values():
            _plot_polylines([ped_xing.edge1.xyz, ped_xing.edge2.xyz], alpha=1.0, color=_LANE_SEGMENT_COLOR)


def _plot_actor_tracks(ax: plt.Axes, scenario: ArgoverseScenario, timestep: int) -> Optional[_PlotBounds]:
    """Plot all actor tracks (up to a particular time step) associated with an Argoverse scenario.

    Args:
        ax: Axes on which actor tracks should be plotted.
        scenario: Argoverse scenario for which to plot actor tracks.
        timestep: Tracks are plotted for all actor data up to the specified time step.

    Returns:
        track_bounds: (x_min, x_max, y_min, y_max) bounds for the extent of actor tracks.
    """
    track_bounds = None
    for track in scenario.tracks:
        # Get timesteps for which actor data is valid
        actor_timesteps: NDArrayInt = np.array(
            [object_state.timestep for object_state in track.object_states if object_state.timestep <= timestep]
        )
        if actor_timesteps.shape[0] < 1 or actor_timesteps[-1] != timestep:
            continue

        # Get actor trajectory and heading history
        actor_trajectory: NDArrayFloat = np.array(
            [list(object_state.position) for object_state in track.object_states if object_state.timestep <= timestep]
        )
        actor_headings: NDArrayFloat = np.array(
            [object_state.heading for object_state in track.object_states if object_state.timestep <= timestep]
        )
        
        # Plot polyline for focal agent location history
        track_color = _DEFAULT_ACTOR_COLOR
        if track.category == TrackCategory.FOCAL_TRACK:
            x_min, x_max = actor_trajectory[:, 0].min(), actor_trajectory[:, 0].max()
            y_min, y_max = actor_trajectory[:, 1].min(), actor_trajectory[:, 1].max()
            track_bounds = (x_min, x_max, y_min, y_max)
            track_color = _FOCAL_AGENT_COLOR
            _plot_polylines([actor_trajectory], color=track_color, line_width=2)
        elif track.track_id == "AV":
            track_color = _AV_COLOR
        elif track.object_type in _STATIC_OBJECT_TYPES:
            continue

        print ('-'*89)
        print(actor_headings[-1])
        # Plot bounding boxes for all vehicles and cyclists
        if track.object_type == ObjectType.VEHICLE:
            
            _plot_actor_bounding_box(
                ax,
                actor_trajectory[-1],
                actor_headings[-1],
                track_color,
                (_ESTIMATED_VEHICLE_LENGTH_M, _ESTIMATED_VEHICLE_WIDTH_M),
            )
        elif track.object_type == ObjectType.CYCLIST or track.object_type == ObjectType.MOTORCYCLIST:
            _plot_actor_bounding_box(
                ax,
                actor_trajectory[-1],
                actor_headings[-1],
                track_color,
                (_ESTIMATED_CYCLIST_LENGTH_M, _ESTIMATED_CYCLIST_WIDTH_M),
            )
        else:
            plt.plot(actor_trajectory[-1, 0], actor_trajectory[-1, 1], "o", color=track_color, markersize=4)

    return track_bounds


def _plot_polylines(
    polylines: Sequence[NDArrayFloat],
    *,
    style: str = "-",
    line_width: float = 1.0,
    alpha: float = 1.0,
    color: str = "r",
) -> None:
    """Plot a group of polylines with the specified config.

    Args:
        polylines: Collection of (N, 2) polylines to plot.
        style: Style of the line to plot (e.g. `-` for solid, `--` for dashed)
        line_width: Desired width for the plotted lines.
        alpha: Desired alpha for the plotted lines.
        color: Desired color for the plotted lines.
    """
    for polyline in polylines:
        plt.plot(polyline[:, 0], polyline[:, 1], style, linewidth=line_width, color=color, alpha=alpha)


def _plot_polygons(polygons: Sequence[NDArrayFloat], *, alpha: float = 1.0, color: str = "r") -> None:
    """Plot a group of filled polygons with the specified config.

    Args:
        polygons: Collection of polygons specified by (N,2) arrays of vertices.
        alpha: Desired alpha for the polygon fill.
        color: Desired color for the polygon.
    """
    for polygon in polygons:
        plt.fill(polygon[:, 0], polygon[:, 1], color=color, alpha=alpha)


def _plot_actor_bounding_box(
    ax: plt.Axes, cur_location: NDArrayFloat, heading: float, color: str, bbox_size: Tuple[float, float]
) -> None:
    """Plot an actor bounding box centered on the actor's current location.

    Args:
        ax: Axes on which actor bounding box should be plotted.
        cur_location: Current location of the actor (2,).
        heading: Current heading of the actor (in radians).
        color: Desired color for the bounding box.
        bbox_size: Desired size for the bounding box (length, width).
    """
    (bbox_length, bbox_width) = bbox_size

    # Compute coordinate for pivot point of bounding box
    d = np.hypot(bbox_length, bbox_width)
    theta_2 = math.atan2(bbox_width, bbox_length)
    pivot_x = cur_location[0] - (d / 2) * math.cos(heading + theta_2)
    pivot_y = cur_location[1] - (d / 2) * math.sin(heading + theta_2)

    vehicle_bounding_box = Rectangle(
        (pivot_x, pivot_y), bbox_length, bbox_width, np.degrees(heading), color=color, zorder=_BOUNDING_BOX_ZORDER
    )
    rect = ax.add_patch(vehicle_bounding_box)
    rect.set_zorder(5)

if __name__ == '__main__':
    scenario, static_map = get_av2_data(split=args.split, idx=20)
    visualize_scenario (scenario, static_map)
