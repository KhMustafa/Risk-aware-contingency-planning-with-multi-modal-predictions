"""Check validity for a given list of trajectories.

Validty Levels:
0 --> Phyical invalid
1 --> Collision in the driven path
2 --> Leaving road boundaries
3 --> Maximum acceptable risk exceeded
10 --> Valid
"""

import numpy as np

from commonroad_dc.collision.trajectory_queries import trajectory_queries
from beliefplanning.planner.utils.timers import ExecTimer
from beliefplanning.planner.Frenet.utils.helper_functions import (
    create_tvobstacle,
)
from beliefplanning.planner.Frenet.utils.prediction_helpers import (
    collision_checker_prediction,
)

VALIDITY_LEVELS = {
    0: "Physically invalid",
    1: "Collision",
    2: "Leaving road boundaries",
    3: "Maximum acceptable risk",
    10: "Valid",
}


def check_validity(
    ft,
    ego_state,
    scenario,
    vehicle_params,
    risk_params,
    mode: str,
    collision_checker,
    predictions: dict = None,
    exec_timer=None,
    start_idx=0,
    mode_num=100,
):
    """
    Check the validity of a frenet trajectory.

    Args:
        ft (FrenetTrajectory): Frenét trajectory to be checked.
        ego_state (State): Current state of the ego vehicle.
        scenario (Scenario): Scenario of the planning problem.
        vehicle_params (VehicleParameters): Parameters of the ego vehicle.
        mode (Str): Mode of the frenét planner.
        road_boundary (ShapeGroup): Shape group of the road boundary.
        collision_checker (CollisionChecker): Collision checker for the considered scenario.
        predictions (dict): Predictions for the visible obstacles. Defaults to None.
        exec_times_dict (dict): Dictionary for the execution times. Defaults to None.

    Returns:
        bool: Valid nor not valid.
        dict: Dictionary with execution times.
    """
    timer = ExecTimer(timing_enabled=False) if exec_timer is None else exec_timer

    with timer.time_with_cm(
        "simulation/sort trajectories/check validity/check velocity"
    ):
        # check maximum velocity
        if not velocity_valid(ft, vehicle_params):
            return 0, "velocity"

    with timer.time_with_cm(
        "simulation/sort trajectories/check validity/check acceleration"
    ):
        # check maximum acceleration:
        if not velocity_valid(ft, vehicle_params):
            return 0, "acceleration"

    with timer.time_with_cm(
        "simulation/sort trajectories/check validity/check curvature"
    ):
        # check maximum curvature

        reason_curvature_invalid = curvature_valid(ft, vehicle_params)
        
        if reason_curvature_invalid is not None:
            return 0, f"curvature ({reason_curvature_invalid})"

    with timer.time_with_cm(
        "simulation/sort trajectories/check validity/check collision"
    ):

        # get collision_object
        collision_object = create_collision_object(ft, vehicle_params, ego_state)

        # collision checkking with obstacles
        '''
        if not collision_valid(
            ft,
            collision_object,
            predictions,
            scenario,
            ego_state,
            collision_checker,
            mode,
            start_idx,
            mode_num
        ):
            return 1, "collision"
        '''
    with timer.time_with_cm(
        "simulation/sort trajectories/check validity/check road boundaries"
    ):
        # check road boundaries
        # if ft.bd_harm is calculated, use it for boundary check
        '''
        if predictions is None:
            if not boundary_valid(vehicle_params, collision_object, road_boundary):
                return 2, "boundaries"
        else:
            if ft.bd_harm:
                return 2, "boundaries"
        '''
    with timer.time_with_cm(
        "simulation/sort trajectories/check validity/check max risk"
    ):
        if not max_risk_valid(ft, risk_params, mode):
            return 3, "max_risk"

    return 10, ""


def create_collision_object(ft, vehicle_params, ego_state):
    """Create a collision_object of the trajectory for collision checking with road boundary and with other vehicles."""
    traj_list = [[ft.x[i], ft.y[i], ft.yaw[i]] for i in range(len(ft.t))]
    collision_object_raw = create_tvobstacle(
        traj_list=traj_list,
        box_length=vehicle_params.l / 2,
        box_width=vehicle_params.w / 2,
        start_time_step=ego_state.time_step,
    )
    # if the preprocessing fails, use the raw trajectory
    collision_object, err = trajectory_queries.trajectory_preprocess_obb_sum(
        collision_object_raw
    )
    if err:
        collision_object = collision_object_raw

    return collision_object


def velocity_valid(ft, vehicle_params):
    """Check if velocity is valid.

    Args:
        ft ([type]): [fretnet trajectory]
        vehicle_params ([type]): [description]

    Returns:
        [bool]: [True if valid, false else]
    """
    # for vi in ft.s_d:
    #     if np.abs(vi) > vehicle_params.longitudinal.v_max:
    #         return False
    # return True
    if np.all(np.abs(ft.s_d) <= vehicle_params.longitudinal.v_max):
        return True
    else:
        return False


def acceleration_valid(ft, vehicle_params):
    """Check if acceleration is valid.

    Args:
        ft ([type]): [fretnet trajectory]
        vehicle_params ([type]): [description]

    Returns:
        [bool]: [True if valid, false else]
    """
    for ai in ft.s_dd:
        if np.abs(ai) > vehicle_params.longitudinal.a_max:
            return False
    return True


def curvature_valid(ft, vehicle_params):
    """Check if acceleration is valid.

    Args:
        ft ([type]): [fretnet trajectory]
        vehicle_params ([type]): [description]

    Returns:
        [bool]: [True if valid, false else]
    """
    # numpy absolute value array operation
    # calculate velocity threshold out of for loop
    # calculate the turning radius
    turning_radius = np.sqrt(
        (vehicle_params.l ** 2 / np.tan(vehicle_params.steering.max) ** 2)
        + (vehicle_params.l_r ** 2)
    )
    # below this velocity, it is assumed that the vehicle can follow the turning radius
    # above this velocity, the curvature is calculated differently
    threshold_low_velocity = np.sqrt(vehicle_params.lateral_a_max * turning_radius)

    c = abs(ft.curv)
    for i in range(len(ft.t)):
        # get curvature via turning radius
        if ft.v[i] < threshold_low_velocity:
            c_max_current, vel_mode = 1.0 / turning_radius, "lvc"
        # get velocity dependent curvature (curvature derived from the lateral acceleration)
        else:
            c_max_current = vehicle_params.lateral_a_max / (ft.v[i] ** 2)
            vel_mode = "hvc"

        if c[i] > (c_max_current):
            return vel_mode
    return None


def boundary_valid(vehicle_params, collision_object, road_boundary):
    """Check if acceleration is valid.

    Args:
        ft ([type]): [fretnet trajectory]
        vehicle_params ([type]): [description]

    Returns:
        [bool]: [True if valid, false else]
    """
    # check if the trajectory leaves the road
    # return an array for the given trajectories (only one trajectory is given)
    # return an integer for every trajectory, at which time step it leaves the road
    # returns -1 if it does not collide with the boundaries
    leaving_road_at = trajectory_queries.trajectories_collision_static_obstacles(
        trajectories=[collision_object],
        static_obstacles=road_boundary,
        method="grid",
        num_cells=32,
        auto_orientation=True,
    )

    if leaving_road_at[0] != -1:
        return False

    return True


def collision_valid(
    ft, collision_object, predictions, scenario, ego_state, collision_checker, mode, start_idx, mode_num
):
    """Check if trajectory is collision free with other predictions.

    Args:
        ft ([type]): [description]
        mode ([type]): [description]

    Returns:
        [type]: [description]
    """
    if mode == "ground_truth":
        collision_detected = collision_checker.collide(collision_object)
        if collision_detected:
            return False

    # check for collision with prediction if predictions are used but not collision_check_prediction, no trajectory
    # is invalid because of collision, they only get collision probabilities
    elif mode == "WaleNet" or mode == "risk":
        collision_detected = collision_checker_prediction(
            predictions=predictions,
            scenario=scenario,
            ego_co=collision_object,
            frenet_traj=ft,
            ego_state=ego_state,
            start_idx=start_idx,
            mode_num=mode_num
        )
        if collision_detected:
            return False

    return True


def max_risk_valid(ft, risk_params, mode):
    """Check for maximum acceptable risk.

    Args:
        ft ([type]): [description]
        risk_params ([type]): [description]

    Returns:
        [type]: [description]
    """
    if mode == "risk":
        if len(ft.obst_risk_dict.values()) > 0:
            if sum(ft.ego_risk_dict.values()) > risk_params["max_acceptable_risk"]:
                return False
            if max(ft.obst_risk_dict.values()) > risk_params["max_acceptable_risk"]:
                return False

    return True
