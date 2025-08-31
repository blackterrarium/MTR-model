# Motion Transformer (MTR): https://arxiv.org/abs/2209.13508
# Published at NeurIPS 2022
# Written by Shaoshuai Shi 
# All Rights Reserved


import sys, os
import numpy as np
import pickle
import tensorflow as tf
import multiprocessing
import glob
from tqdm import tqdm
from waymo_open_dataset.protos import scenario_pb2
from waymo_types import object_type, lane_type, road_line_type, road_edge_type, signal_state, polyline_type

    
def decode_tracks_from_proto(tracks):
    track_infos = {
        'object_id': [],  # {0: unset, 1: vehicle, 2: pedestrian, 3: cyclist, 4: others}
        'object_type': [],
        'trajs': []
    }
    for i, cur_data in enumerate(tracks):
        cur_traj = [np.array([x.center_x, x.center_y, x.center_z, x.length, x.width, x.height, x.heading,
                            x.velocity_x, x.velocity_y, x.valid], dtype=np.float32) for x in cur_data.states]
        cur_traj = np.stack(cur_traj, axis=0)  # (num_timestamp, 10)

        track_infos['object_id'].append(cur_data.id)
        track_infos['object_type'].append(object_type[cur_data.object_type])
        track_infos['trajs'].append(cur_traj)
        
    track_infos['trajs'] = np.stack(track_infos['trajs'], axis=0)  # (num_objects, num_timestamp, 9)
    return track_infos


def get_polyline_dir(polyline):
    polyline_pre = np.roll(polyline, shift=1, axis=0)
    polyline_pre[0] = polyline[0]
    diff = polyline - polyline_pre
    polyline_dir = diff / np.clip(np.linalg.norm(diff, axis=-1)[:, np.newaxis], a_min=1e-6, a_max=1000000000)
    return polyline_dir


def decode_map_features_from_proto(map_features):
    map_infos = {
        'lane': [],
        'road_line': [],
        'road_edge': [],
        'stop_sign': [],
        'crosswalk': [],
        'speed_bump': []
    }
    polylines = []

    point_cnt = 0
    for cur_data in map_features:
        cur_info = {'id': cur_data.id}

        if cur_data.lane.ByteSize() > 0:
            cur_info['speed_limit_mph'] = cur_data.lane.speed_limit_mph
            cur_info['type'] = lane_type[cur_data.lane.type]  # 0: undefined, 1: freeway, 2: surface_street, 3: bike_lane

            cur_info['interpolating'] = cur_data.lane.interpolating
            cur_info['entry_lanes'] = list(cur_data.lane.entry_lanes)
            cur_info['exit_lanes'] = list(cur_data.lane.exit_lanes)

            cur_info['left_boundary'] = [{
                    'start_index': x.lane_start_index, 'end_index': x.lane_end_index,
                    'feature_id': x.boundary_feature_id,
                    'boundary_type': x.boundary_type  # roadline type
                } for x in cur_data.lane.left_boundaries
            ]
            cur_info['right_boundary'] = [{
                    'start_index': x.lane_start_index, 'end_index': x.lane_end_index,
                    'feature_id': x.boundary_feature_id,
                    'boundary_type': road_line_type[x.boundary_type]  # roadline type
                } for x in cur_data.lane.right_boundaries
            ]

            global_type = polyline_type[cur_info['type']]
            cur_polyline = np.stack([np.array([point.x, point.y, point.z, global_type]) for point in cur_data.lane.polyline], axis=0)
            cur_polyline_dir = get_polyline_dir(cur_polyline[:, 0:3])
            cur_polyline = np.concatenate((cur_polyline[:, 0:3], cur_polyline_dir, cur_polyline[:, 3:]), axis=-1)

            map_infos['lane'].append(cur_info)

        elif cur_data.road_line.ByteSize() > 0:
            cur_info['type'] = road_line_type[cur_data.road_line.type]

            global_type = polyline_type[cur_info['type']]
            cur_polyline = np.stack([np.array([point.x, point.y, point.z, global_type]) for point in cur_data.road_line.polyline], axis=0)
            cur_polyline_dir = get_polyline_dir(cur_polyline[:, 0:3])
            cur_polyline = np.concatenate((cur_polyline[:, 0:3], cur_polyline_dir, cur_polyline[:, 3:]), axis=-1)

            map_infos['road_line'].append(cur_info)

        elif cur_data.road_edge.ByteSize() > 0:
            cur_info['type'] = road_edge_type[cur_data.road_edge.type]

            global_type = polyline_type[cur_info['type']]
            cur_polyline = np.stack([np.array([point.x, point.y, point.z, global_type]) for point in cur_data.road_edge.polyline], axis=0)
            cur_polyline_dir = get_polyline_dir(cur_polyline[:, 0:3])
            cur_polyline = np.concatenate((cur_polyline[:, 0:3], cur_polyline_dir, cur_polyline[:, 3:]), axis=-1)

            map_infos['road_edge'].append(cur_info)

        elif cur_data.stop_sign.ByteSize() > 0:
            cur_info['lane_ids'] = list(cur_data.stop_sign.lane)
            point = cur_data.stop_sign.position
            cur_info['position'] = np.array([point.x, point.y, point.z])

            global_type = polyline_type['TYPE_STOP_SIGN']
            cur_polyline = np.array([point.x, point.y, point.z, 0, 0, 0, global_type]).reshape(1, 7)

            map_infos['stop_sign'].append(cur_info)
        elif cur_data.crosswalk.ByteSize() > 0:
            global_type = polyline_type['TYPE_CROSSWALK']
            cur_polyline = np.stack([np.array([point.x, point.y, point.z, global_type]) for point in cur_data.crosswalk.polygon], axis=0)
            cur_polyline_dir = get_polyline_dir(cur_polyline[:, 0:3])
            cur_polyline = np.concatenate((cur_polyline[:, 0:3], cur_polyline_dir, cur_polyline[:, 3:]), axis=-1)

            map_infos['crosswalk'].append(cur_info)

        elif cur_data.speed_bump.ByteSize() > 0:
            global_type = polyline_type['TYPE_SPEED_BUMP']
            cur_polyline = np.stack([np.array([point.x, point.y, point.z, global_type]) for point in cur_data.speed_bump.polygon], axis=0)
            cur_polyline_dir = get_polyline_dir(cur_polyline[:, 0:3])
            cur_polyline = np.concatenate((cur_polyline[:, 0:3], cur_polyline_dir, cur_polyline[:, 3:]), axis=-1)

            map_infos['speed_bump'].append(cur_info)

        else:
            print(cur_data)
            raise ValueError

        polylines.append(cur_polyline)
        cur_info['polyline_index'] = (point_cnt, point_cnt + len(cur_polyline))
        point_cnt += len(cur_polyline)

    try:
        polylines = np.concatenate(polylines, axis=0).astype(np.float32)
    except:
        polylines = np.zeros((0, 7), dtype=np.float32)
        print('Empty polylines: ')
    map_infos['all_polylines'] = polylines
    return map_infos


def decode_dynamic_map_states_from_proto(dynamic_map_states):
    dynamic_map_infos = {
        'lane_id': [],
        'state': [],
        'stop_point': []
    }
    for cur_data in dynamic_map_states:  # (num_timestamp)
        lane_id, state, stop_point = [], [], []
        for cur_signal in cur_data.lane_states:  # (num_observed_signals)
            lane_id.append(cur_signal.lane)
            state.append(signal_state[cur_signal.state])
            stop_point.append([cur_signal.stop_point.x, cur_signal.stop_point.y, cur_signal.stop_point.z])

        dynamic_map_infos['lane_id'].append(np.array([lane_id]))
        dynamic_map_infos['state'].append(np.array([state]))
        dynamic_map_infos['stop_point'].append(np.array([stop_point]))

    return dynamic_map_infos

def filter_tracks_by_distance(track_infos, info, threshold=10):
    """
    Filter tracks based on their distance from a center point.
    :param track_infos: Dictionary containing track information.
    :param center_index: Index of the center track.
    :param threshold: Distance threshold for filtering.
    :return: Filtered track information.
    """
    # Extract trajectories from track_infos
    trajectories = track_infos['trajs']  

    center_index =0 
    for i, index in enumerate(info['tracks_to_predict']['track_index']):
        if info['tracks_to_predict']['object_type'][i]=='TYPE_VEHICLE':
            center_index = index
            break

    # Find the last valid timestamp for past trajectories
    past_trajectories = trajectories[:, :11, :]
    valid_flags = past_trajectories[:, :, -1]  # shape: (196, 91)
    last_valid_idx = (valid_flags * np.arange(11)).max(axis=1).astype(int)  

    # get the last valid position of center trajectory and filtered trajectorie
    center_traj_last_valid_pos = trajectories[center_index,last_valid_idx[center_index],:3]
    other_trajs_last_valid_pos = trajectories[np.arange(len(trajectories)), last_valid_idx, :3]

    # Calculate distances from the center trajectory's last valid position
    distances = np.linalg.norm(other_trajs_last_valid_pos - center_traj_last_valid_pos, axis=1)  # shape (107,)


    mask_beyond_threshold = distances > threshold

    # Zero out those trajectories
    trajectories[mask_beyond_threshold] = 0

    # Count how many are within the threshold
    count_within_threshold = np.sum(~mask_beyond_threshold)

    print('Number of surrounding agents: ', count_within_threshold)

    # mask = last_valid_idx != 0
    # # Filter trajectories for whihc there is no valid past trajectory
    # filtered_trajectories = trajectories[mask] 
    # filtered_object_types = np.array(object_types)[mask]
    # filtered_object_ids = np.array(object_ids)[mask]
    # past_filtered_trajectories = filtered_trajectories[:, :11, :]

    # # Get the index of latest valid timestamp for filtered past trajectories
    # # as the size of array cahnges after filtering, thats why we need to recalculate
    # # last_valid_idx for filtered trajectories
    # valid_flags_filtered_trajectories = past_filtered_trajectories[:, :, -1] 
    # last_valid_idx_filtered_trajectories = (valid_flags_filtered_trajectories * np.arange(11)).max(axis=1).astype(int)  


    # # get the index of first vehicle track to predict
    # center_index = 0
    # center_difficulty = 0
    # center_object_type = None
    # for i, object_type in enumerate(info['tracks_to_predict']['object_type']):
    #     if object_type=='TYPE_VEHICLE':
    #         center_object_type= object_type
    #         center_index = i
    #         center_difficulty = info['tracks_to_predict']['difficulty'][i]
    #         break
    
    # # get the last valid position of center trajectory and filtered trajectorie
    # center_traj_last_valid_pos = trajectories[center_index,last_valid_idx[center_index],:3]
    # other_trajs_last_valid_pos = filtered_trajectories[np.arange(len(filtered_trajectories)), last_valid_idx_filtered_trajectories, :3]

    # # Calculate distances from the center trajectory's last valid position
    # distances = np.linalg.norm(other_trajs_last_valid_pos - center_traj_last_valid_pos, axis=1)  # shape (107,)

    # # Step 2: Filter points where distance <= threshold
    # filtered_trajectories = filtered_trajectories[distances <= threshold]
    # filtered_object_types = filtered_object_types[distances <= threshold]
    # filtered_object_ids = filtered_object_ids[distances <= threshold]

    # # find the index of center trajectory in filtered trajectories
    # for i,  traj in enumerate(filtered_trajectories):
    #     if (np.linalg.norm(traj[last_valid_idx[center_index]][:3]-center_traj_last_valid_pos ) ==0):
    #         center_index = i


    # # pad the filtered trajectories to match the original number of trajectories
    # # if not matched then cause error like sdc_track_index out of range
    # filtered_trajectories = np.pad(filtered_trajectories, ((0, len(trajectories) - len(filtered_trajectories)), (0, 0), (0, 0)), mode='constant', constant_values=0)
    # filtered_object_types = np.pad(filtered_object_types, (0, len(trajectories) - len(filtered_trajectories)), mode='constant', constant_values='TYPE_UNSET')
    # filtered_object_ids = np.pad(filtered_object_ids, (0, len(trajectories) - len(filtered_trajectories)), mode='constant', constant_values=0)


    return trajectories ,  center_index

def process_waymo_data_with_scenario_proto(data_file, output_path=None):
    dataset = tf.data.TFRecordDataset(data_file, compression_type='')
    ret_infos = []
    for cnt, data in enumerate(dataset):
        info = {}
        scenario = scenario_pb2.Scenario()
        scenario.ParseFromString(bytes(data.numpy()))
        if scenario.scenario_id == 'cf372a7f36ed5b3c': # debuging single example with 196 agents

            info['scenario_id'] = scenario.scenario_id
            info['timestamps_seconds'] = list(scenario.timestamps_seconds)  # list of int of shape (91)
            info['current_time_index'] = scenario.current_time_index  # int, 10
            info['sdc_track_index'] = scenario.sdc_track_index  # int
            info['objects_of_interest'] = list(scenario.objects_of_interest)  # list, could be empty list

            info['tracks_to_predict'] = {
                'track_index': [cur_pred.track_index for cur_pred in scenario.tracks_to_predict],
                'difficulty': [cur_pred.difficulty for cur_pred in scenario.tracks_to_predict]
            }  # for training: suggestion of objects to train on, for val/test: need to be predicted

            track_infos = decode_tracks_from_proto(scenario.tracks)
            info['tracks_to_predict']['object_type'] = [track_infos['object_type'][cur_idx] for cur_idx in info['tracks_to_predict']['track_index']]
            
            trajectories , center_index  = filter_tracks_by_distance(track_infos, info, threshold=30)
            
            track_infos['trajs'] = trajectories
            info['tracks_to_predict']['track_index'] = [center_index]
            info['tracks_to_predict']['difficulty'] = [cur_pred.difficulty for cur_pred in scenario.tracks_to_predict if cur_pred.track_index == center_index]
            info['tracks_to_predict']['object_type'] = [track_infos['object_type'][cur_idx] for cur_idx in info['tracks_to_predict']['track_index']]


            # decode map related data
            map_infos = decode_map_features_from_proto(scenario.map_features)
            dynamic_map_infos = decode_dynamic_map_states_from_proto(scenario.dynamic_map_states)

            save_infos = {
                'track_infos': track_infos,
                'dynamic_map_infos': dynamic_map_infos,
                'map_infos': map_infos
            }
            save_infos.update(info)

            output_file = os.path.join(output_path, f'sample_{scenario.scenario_id}.pkl')
            with open(output_file, 'wb') as f:
                pickle.dump(save_infos, f)

            ret_infos.append(info)


    return ret_infos


def get_infos_from_protos(data_path, output_path=None, num_workers=8):
    from functools import partial
    if output_path is not None:
        os.makedirs(output_path, exist_ok=True)

    func = partial(
        process_waymo_data_with_scenario_proto, output_path=output_path
    )

    src_files = glob.glob(os.path.join(data_path, '*.tfrecord*'))
    src_files.sort()

    # func(src_files[0])
    with multiprocessing.Pool(num_workers) as p:
        data_infos = list(tqdm(p.imap(func, src_files), total=len(src_files)))

    all_infos = [item for infos in data_infos for item in infos]
    return all_infos


def create_infos_from_protos(raw_data_path, output_path, num_workers=16):
    # train_infos = get_infos_from_protos(
    #     data_path=os.path.join(raw_data_path, 'training'),
    #     output_path=os.path.join(output_path, 'processed_scenarios_training'),
    #     num_workers=num_workers
    # )
    # train_filename = os.path.join(output_path, 'processed_scenarios_training_infos.pkl')
    # with open(train_filename, 'wb') as f:
    #     pickle.dump(train_infos, f)
    # print('----------------Waymo info train file is saved to %s----------------' % train_filename)

    # val_infos = get_infos_from_protos(
    #     data_path=os.path.join(raw_data_path, 'validation'),
    #     output_path=os.path.join(output_path, 'processed_scenarios_validation'),
    #     num_workers=num_workers
    # )
    # val_filename = os.path.join(output_path, 'processed_scenarios_val_infos.pkl')
    # with open(val_filename, 'wb') as f:
    #     pickle.dump(val_infos, f)
    # print('----------------Waymo info val file is saved to %s----------------' % val_filename)

    # debug_infos = get_infos_from_protos(
    #     data_path=os.path.join(raw_data_path, 'debuging'),
    #     output_path=os.path.join(output_path, 'processed_scenarios_debuging_single_example_30m'),
    #     num_workers=num_workers
    # )
    # debug_filename = os.path.join(output_path, 'processed_scenarios_debuging_single_example_30m_infos.pkl')
    # with open(debug_filename, 'wb') as f:
    #     pickle.dump(debug_infos, f)
    # print('----------------Waymo info val file is saved to %s----------------' % debug_filename)

    val_infos = get_infos_from_protos(
        data_path=os.path.join(raw_data_path, 'validation_interactive'),
        output_path=os.path.join(output_path, 'processed_scenarios_validation_interactive'),
        num_workers=num_workers
    )
    val_filename = os.path.join(output_path, 'processed_scenarios_val_inter_infos.pkl')
    with open(val_filename, 'wb') as f:
        pickle.dump(val_infos, f)
    print('----------------Waymo info val file is saved to %s----------------' % val_filename)
    
    

if __name__ == '__main__':
    create_infos_from_protos(
        raw_data_path=sys.argv[1],
        output_path=sys.argv[2]
    )
