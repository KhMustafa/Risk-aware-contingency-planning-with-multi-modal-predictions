#!/user/bin/env python

"""Helper functions to adjust the prediction to the needs of the frenét planner."""

import numpy as np
import os
import sys
from commonroad.scenario.obstacle import ObstacleRole
from commonroad_dc.collision.trajectory_queries import trajectory_queries
import torch
from torch.distributions.normal import Normal

module_path = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
sys.path.append(module_path)

from planner.Frenet.utils.helper_functions import create_tvobstacle, distance


def belief_updater(predictions: dict, belief):
    prob = {}
    for obstacle_id in list(predictions.keys()):
        obstacle_orientation_state = torch.tensor(predictions[obstacle_id]['orientation_list'][0][0])

        for mode in range(len(predictions[obstacle_id]['orientation_list'])):
            mu = torch.tensor(predictions[obstacle_id]['orientation_list'][mode][0])
            dist = Normal(mu, 1.)
            prob[mode] = torch.exp(dist.log_prob(obstacle_orientation_state))

        belief[0] = prob[0] * belief[0] / (prob[0] * belief[0] + prob[1] * belief[1])
        belief[1] = 1 - belief[0]
        # transform tensor to floating point.
        belief[0] = belief[0].item()
        belief[1] = belief[1].item()

    return belief


def get_obstacles_in_radius(scenario, ego_id: int, ego_state, radius: float):
    """
    Get all the obstacles that can be found in a given radius.

    Args:
        scenario (Scenario): Considered Scenario.
        ego_id (int): ID of the ego vehicle.
        ego_state (State): State of the ego vehicle.
        radius (float): Considered radius.

    Returns:
        [int]: List with the IDs of the obstacles that can be found in the ball with the given radius centering at the ego vehicles position.
    """
    obstacles_within_radius = []
    for obstacle in scenario.obstacles:
        # do not consider the ego vehicle
        if obstacle.obstacle_id != ego_id:
            occ = obstacle.occupancy_at_time(ego_state.time_step)
            # if the obstacle is not in the lanelet network at the given time, its occupancy is None
            if occ is not None:
                # calculate the distance between the two obstacles
                if type(occ) == list:
                    dist = distance(
                        pos1=ego_state.position,
                        pos2=obstacle.occupancy_at_time(ego_state.time_step)[0].shape.center,
                    )
                else:
                    dist = distance(
                        pos1=ego_state.position,
                        pos2=obstacle.occupancy_at_time(ego_state.time_step).shape.center,
                    )

                # add obstacles that are close enough
                if dist < radius:
                    obstacles_within_radius.append(obstacle.obstacle_id)

    return obstacles_within_radius


def get_dyn_and_stat_obstacles(obstacle_ids: [int], scenario):
    """
    Split a set of obstacles in a set of dynamic obstacles and a set of static obstacles.

    Args:
        obstacle_ids ([int]): IDs of all considered obstacles.
        scenario: Considered scenario.

    Returns:
        [int]: List with the IDs of all dynamic obstacles.
        [int]: List with the IDs of all static obstacles.

    """
    dyn_obstacles = []
    stat_obstacles = []
    for obst_id in obstacle_ids:
        if scenario.obstacle_by_id(obst_id).obstacle_role == ObstacleRole.DYNAMIC:
            dyn_obstacles.append(obst_id)
        else:
            stat_obstacles.append(obst_id)

    return dyn_obstacles, stat_obstacles


def get_orientation_velocity_and_shape_of_prediction(
        predictions: dict, scenario, safety_margin_length=1.0, safety_margin_width=0.5
):
    """
    Extend the prediction by adding information about the orientation, velocity and the shape of the predicted obstacle.

    Args:
        predictions (dict): Prediction dictionary that should be extended.
        scenario (Scenario): Considered scenario.

    Returns:
        dict: Extended prediction dictionary.
        :param safety_margin_width:
        :param predictions:
        :param scenario:
        :param safety_margin_length:
    """
    # go through every predicted obstacle
    obstacle_ids = list(predictions.keys())
    for obstacle_id in obstacle_ids:
        obstacle = scenario.obstacle_by_id(obstacle_id)
        # get x- and y-position of the predicted trajectory
        pred_traj = predictions[obstacle_id]['pos_list']

        # added by Khaled. Convert the 2D array into a list of 2D array, to use the same routine
        # for both uni-modal and multi-modal predictions.
        # not needed anymore, the if condition can be removed, and only the else part should be kept
        if type(pred_traj) != list:
            pred_traj_list_tmp = list()
            for entry in pred_traj:
                pred_traj_list_tmp.append(entry)
            pred_traj_list = list()
            pred_traj_list.append(np.array(pred_traj_list_tmp))
        else:
            pred_traj_list = pred_traj
            pred_orientation_list = [[None]] * len(obstacle.prediction)
            pred_v_list = [[None]] * len(obstacle.prediction)

        index = 0
        for pred_traj in pred_traj_list:
            pred_length = len(pred_traj)

            # there may be some predictions without any trajectory (when the obstacle disappears due to exceeding time)
            if pred_length == 0:
                del predictions[obstacle_id]
                continue

            # for predictions with only one timestep, the gradient can not be derived --> use initial orientation
            if pred_length == 1:
                pred_orientation = [obstacle.initial_state.orientation]
                pred_v = [obstacle.initial_state.velocity]
            else:
                t = [0.0 + i * scenario.dt for i in range(pred_length)]
                x = pred_traj[:, 0][0:pred_length]
                y = pred_traj[:, 1][0:pred_length]

                # calculate the yaw angle for the predicted trajectory
                dx = np.gradient(x, t)
                dy = np.gradient(y, t)
                # if the vehicle does barely move, use the initial orientation
                # otherwise small uncertainties in the position can lead to great orientation uncertainties
                if all(dxi < 0.0001 for dxi in dx) and all(dyi < 0.0001 for dyi in dy):
                    init_orientation = obstacle.initial_state.orientation
                    pred_orientation = np.full((1, pred_length), init_orientation)[0]
                # if the vehicle moves, calculate the orientation
                else:
                    pred_orientation = np.arctan2(dy, dx)

                # get the velocity from the derivation of the position
                pred_v = np.sqrt((np.power(dx, 2) + np.power(dy, 2)))

                # add the new information to the prediction dictionary
                pred_orientation_list[index] = pred_orientation
                pred_v_list[index] = pred_v
                index = index + 1
                '''
                    predictions[obstacle_id]['orientation_list'] = pred_orientation
                    predictions[obstacle_id]['v_list'] = pred_v
                    obstacle_shape = obstacle.obstacle_shape
                    predictions[obstacle_id]['shape'] = {
                        'length': obstacle_shape.length + safety_margin_length,
                        'width': obstacle_shape.width + safety_margin_width,
                    }
                '''
        # this to make the orientation_list as a list of different modes. If the prediction
        # has a single mode, the list will have a single entry.
        predictions[obstacle_id]['orientation_list'] = pred_orientation_list
        predictions[obstacle_id]['v_list'] = pred_v_list
        obstacle_shape = obstacle.obstacle_shape
        predictions[obstacle_id]['shape'] = {
            'length': obstacle_shape.length + safety_margin_length,
            'width': obstacle_shape.width + safety_margin_width,
        }

    # return the updated predictions dictionary
    return predictions


def collision_checker_prediction(
        predictions: dict, scenario, ego_co, frenet_traj, ego_state, start_idx, mode_num
):
    """
    Check predictions for collisions.

    Args:
        predictions (dict): Dictionary with the predictions of the obstacles.
        scenario (Scenario): Considered scenario.
        ego_co (TimeVariantCollisionObject): The collision object of the ego vehicles trajectory.
        frenet_traj (FrenetTrajectory): Considered trajectory.
        ego_state (State): Current state of the ego vehicle.

    Returns:
        bool: True if the trajectory collides with a prediction.
    """
    # check every obstacle in the predictions
    for obstacle_id in list(predictions.keys()):
        for mode in range(len(predictions[obstacle_id]['pos_list'])):
            if mode_num != 2:
                mode = mode_num
            # check if the obstacle is not a rectangle (only shape with attribute length)
            if not hasattr(scenario.obstacle_by_id(obstacle_id).obstacle_shape, 'length'):
                raise Warning('Collision Checker can only handle rectangular obstacles.')
            else:

                # get dimensions of the obstacle
                length = predictions[obstacle_id]['shape']['length']
                width = predictions[obstacle_id]['shape']['width']

                # only check for collision as long as both trajectories (frenét trajectory and prediction) are visible
                pred_traj = predictions[obstacle_id]['pos_list'][mode][start_idx:]
                pred_length = min(len(frenet_traj.t), len(pred_traj))
                if pred_length == 0:
                    continue

                # get x, y and orientation of the prediction
                x = pred_traj[:, 0][0:pred_length]
                y = pred_traj[:, 1][0:pred_length]
                pred_orientation = predictions[obstacle_id]['orientation_list'][mode][start_idx:]

                # create a time variant collision object for the predicted vehicle
                traj = [[x[i], y[i], pred_orientation[i]] for i in range(pred_length)]

                prediction_collision_object_raw = create_tvobstacle(
                    traj_list=traj,
                    box_length=length / 2,
                    box_width=width / 2,
                    start_time_step=ego_state.time_step + 1,
                )

            # preprocess the collision object
            # if the preprocessing fails, use the raw trajectory
            (
                prediction_collision_object,
                err,
            ) = trajectory_queries.trajectory_preprocess_obb_sum(
                prediction_collision_object_raw
            )
            if err:
                prediction_collision_object = prediction_collision_object_raw

            # check for collision between the trajectory of the ego obstacle and the predicted obstacle
            collision_at = trajectory_queries.trajectories_collision_dynamic_obstacles(
                trajectories=[ego_co],
                dynamic_obstacles=[prediction_collision_object],
                method='grid',
                num_cells=32,
            )

            # if there is no collision (returns -1) return False, else True
            if collision_at[0] != -1:
                return True

            if mode_num != 2:
                break

    return False


def add_static_obstacle_to_prediction(
        predictions: dict, obstacle_id_list: [int], scenario, pred_horizon: int = 50
):
    """
    Add static obstacles to the prediction since predictor can not handle static obstacles.

    Args:
        predictions (dict): Dictionary with the predictions.
        obstacle_id_list ([int]): List with the IDs of the static obstacles.
        scenario (Scenario): Considered scenario.
        pred_horizon (int): Considered prediction horizon. Defaults to 50.

    Returns:
        dict: Dictionary with the predictions.
    """
    for obstacle_id in obstacle_id_list:
        obstacle = scenario.obstacle_by_id(obstacle_id)
        fut_pos = []
        fut_cov = []
        # create a mean and covariance matrix for every time step in the prediction horizon
        for ts in range(int(pred_horizon)):
            fut_pos.append(obstacle.initial_state.position)
            fut_cov.append([[0.02, 0.0], [0.0, 0.02]])

        fut_pos = np.array(fut_pos)
        fut_cov = np.array(fut_cov)

        # add the prediction to the prediction dictionary
        predictions[obstacle_id] = {'pos_list': fut_pos, 'cov_list': fut_cov}

    return predictions


def get_obstacles_prediction_overtake(zPred, backup):
    prediction_result = {}
    init_state = zPred[0][0]
    obst_length = 6
    obst_width = 2.5
    n = 4
    m = 3  # number of policies
    fut_pos_list = [[None]] * m
    fut_cov_list = [[None]] * m
    pred_orientation_list = [[None]] * m
    pred_v_list = [[None]] * m
    fut_pos = []
    fut_cov = []

    for ts in range(backup[1].shape[0] + 1):
        for mode in range(3):
            if ts == 0:
                fut_pos_list[mode] = list()
                fut_cov_list[mode] = list()
                fut_pos_list[mode].append(init_state[0:2])
                fut_cov_list[mode].append([[0.1, 0.0], [0.0, 0.1]])
            else:
                fut_pos_list[mode].append(backup[1][ts - 1, mode * n: (mode * n + 2)])
                fut_cov_list[mode].append([[0.1, 0.0], [0.0, 0.1]])

    # add orientation and velocity to the prediction dict
    for mode in range(m):
        pred_orientation_list[mode] = backup[1][:, n * mode + 3]
        pred_v_list[mode] = backup[1][:, n * mode + 2]

    for mode in range(m):
        pred_orientation_list[mode] = np.insert(pred_orientation_list[mode], 0, init_state[3])
        pred_v_list[mode] = np.insert(pred_v_list[mode], 0, init_state[2])

    if len(fut_pos) == 0:
        fut_pos = fut_pos_list
        fut_cov = fut_cov_list
        for i in range(len(fut_pos)):
            fut_pos[i] = np.array(fut_pos[i])
            fut_cov[i] = np.array(fut_cov[i])

    prediction_result[1] = {'pos_list': fut_pos, 'cov_list': fut_cov, 'orientation_list': pred_orientation_list,
                            'v_list': pred_v_list}
    prediction_result[1]['shape'] = {'length': obst_length, 'width': obst_width}
    return prediction_result


def get_prediction_from_scenario_tree(zPred):
    prediction_result = {}
    m = 3
    scenarios_list = []

    for mode_index in range(m):
        for i in range(mode_index * m + m, mode_index * m + 2 * m):
            scenarios_list.append(np.concatenate((zPred[mode_index][:-1], zPred[i])))

    fut_pos_list = [[None]] * 9
    fut_cov_list = [[None]] * 9
    pred_orientation_list = [[None]] * 9
    pred_v_list = [[None]] * 9
    fut_pos = []
    fut_cov = []
    obst_length = 6
    obst_width = 2.6

    for ts in range(scenarios_list[0].shape[0]):
        for mode in range(m * m):
            if ts == 0:
                fut_pos_list[mode] = list()
                fut_cov_list[mode] = list()
                fut_pos_list[mode].append(scenarios_list[mode][ts][0:2])
                fut_cov_list[mode].append([[0.1, 0.0], [0.0, 0.1]])
            else:
                fut_pos_list[mode].append(scenarios_list[mode][ts][0:2])
                fut_cov_list[mode].append([[0.1, 0.0], [0.0, 0.1]])

    # add orientation and velocity to the prediction dict
    for mode in range(m * m):
        pred_orientation_list[mode] = scenarios_list[mode][:, 3]
        pred_v_list[mode] = scenarios_list[mode][:, 2]
    '''
    for mode in range(m):
        pred_orientation_list[mode] = np.insert(pred_orientation_list[mode], 0, init_state[3])
        pred_v_list[mode] = np.insert(pred_v_list[mode], 0, init_state[2])
    '''
    if len(fut_pos) == 0:
        fut_pos = fut_pos_list
        fut_cov = fut_cov_list
        for i in range(len(fut_pos)):
            fut_pos[i] = np.array(fut_pos[i])
            fut_cov[i] = np.array(fut_cov[i])

    prediction_result[1] = {'pos_list': fut_pos, 'cov_list': fut_cov, 'orientation_list': pred_orientation_list,
                            'v_list': pred_v_list}
    prediction_result[1]['shape'] = {'length': obst_length, 'width': obst_width}
    return prediction_result


def get_ground_truth_prediction(
        obstacle_ids: [int], scenario, time_step: int, pred_horizon: int = 100
):
    """
    Transform the ground truth to a prediction. Use this if the prediction fails.

    Args:
        obstacle_ids ([int]): IDs of the visible obstacles.
        scenario (Scenario): considered scenario.
        time_step (int): Current time step.
        pred_horizon (int): Prediction horizon for the prediction.

    Returns:
        dict: Dictionary with the predictions.
    """
    # create a dictionary for the predictions
    prediction_result = {}
    for obstacle_id in obstacle_ids:
        obstacle = scenario.obstacle_by_id(obstacle_id)
        fut_pos = []
        fut_cov = []
        # predefine the length of the lists. It should be the same as number of prediction modes
        fut_pos_list = [[None]] * len(obstacle.prediction)
        fut_cov_list = [[None]] * len(obstacle.prediction)
        # predict dynamic obstacles as long as they are in the scenario
        if obstacle.obstacle_role == ObstacleRole.DYNAMIC:
            # The "list" check is added by Khaled
            if type(obstacle.prediction) == list:
                len_pred = len(obstacle.prediction[0].occupancy_set)
            else:
                len_pred = len(obstacle.prediction.occupancy_set)

        # predict static obstacles for the length of the prediction horizon
        else:
            len_pred = pred_horizon
        # create mean and the covariance matrix of the obstacles
        # for ts in range(time_step, min(pred_horizon, len_pred)):
        for ts in range(time_step, min(time_step + pred_horizon, len_pred)):
            # get the occupancy of an obstacles (if it is not in the scenario at the given time step, the occupancy
            # is None)
            occupancy = obstacle.occupancy_at_time(ts)
            if occupancy is not None:
                # create mean and covariance matrix
                if type(occupancy) == list:
                    for index in range(len(obstacle.prediction)):
                        if ts == time_step:
                            fut_pos_list[index] = list()
                            fut_cov_list[index] = list()
                            fut_pos_list[index].append(occupancy[index].shape.center)
                            fut_cov_list[index].append([[0.1 + (ts * 0.1) * 0.1, 0.0], [0.0, 0.1 + (ts * 0.1) * 0.1]])
                        else:
                            fut_pos_list[index].append(occupancy[index].shape.center)
                            fut_cov_list[index].append([[0.1 + (ts * 0.1) * 0.1, 0.0], [0.0, 0.1 + (ts * 0.1) * 0.1]])
                else:
                    if type(obstacle.prediction) == list and len(obstacle.prediction) > 1:
                        for index in range(len(obstacle.prediction)):
                            fut_pos_list[index] = list()
                            fut_cov_list[index] = list()
                            fut_pos_list[index].append(occupancy.shape.center)
                            fut_cov_list[index].append([[0.1, 0.0], [0.0, 0.1]])
                    else:
                        fut_pos.append(occupancy.shape.center)
                        fut_cov.append([[0.1, 0.0], [0.0, 0.1]])

        if len(fut_pos) == 0:
            fut_pos = fut_pos_list
            fut_cov = fut_cov_list
            for i in range(len(fut_pos)):
                fut_pos[i] = np.array(fut_pos[i])
                fut_cov[i] = np.array(fut_cov[i])
        else:
            fut_pos = np.array(fut_pos)
            fut_cov = np.array(fut_cov)
            fut_pos_list = list()
            fut_cov_list = list()
            fut_pos_list.append(fut_pos)
            fut_cov_list.append(fut_cov)
            fut_pos = fut_pos_list
            fut_cov = fut_cov_list

        # add the prediction for the considered obstacle
        prediction_result[obstacle_id] = {'pos_list': fut_pos, 'cov_list': fut_cov}

    return prediction_result

# EOF
