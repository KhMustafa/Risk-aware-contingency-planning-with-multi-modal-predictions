"""Risk cost function and principles of ethics of risk."""

from beliefplanning.planner.Frenet.utils.validity_checks import create_collision_object
from commonroad_dc.collision.trajectory_queries import trajectory_queries

from beliefplanning.risk_assessment.harm_estimation import get_harm
from beliefplanning.risk_assessment.collision_probability import (
    get_collision_probability_fast,
    get_inv_mahalanobis_dist
)
from beliefplanning.risk_assessment.helpers.timers import ExecTimer
from beliefplanning.planner.utils.responsibility import assign_responsibility_by_action_space, \
    calc_responsibility_reach_set
from beliefplanning.risk_assessment.utils.logistic_regression_symmetrical import \
    get_protected_inj_prob_log_reg_ignore_angle


def calc_risk(
        traj,
        ego_state,
        predictions: dict,
        scenario,
        ego_id: int,
        vehicle_params,
        params,
        reach_set=None,
        exec_timer=None,
        start_idx=0,
        mode_num=100,
        belief=None,
):
    """
    Calculate the risk for the given trajectory.

    Args:
        traj (FrenetTrajectory): Considered frenét trajectory.
        predictions (dict): Predictions for the visible obstacles.
        scenario (Scenario): Considered scenario.
        ego_id (int): ID of the ego vehicle.
        vehicle_params (VehicleParameters): Vehicle parameters of the
            considered vehicle.
        weights (Dict): Weighing factors. Read from weights.json.
        modes (Dict): Risk modes. Read from risk.json.
        coeffs (Dict): Risk parameters. Read from risk_parameters.json.
        exec_timer (ExecTimer): Timer for the exec_timing.json.

    Returns:
        float: Weighed risk costs.
        dict: Dictionary with ego harms for every time step concerning every
            obstacle
        dict: Dictionary with obstacle harms for every time step concerning
            every obstacle

    """
    timer = ExecTimer(timing_enabled=False) if exec_timer is None else exec_timer

    modes = params['modes']
    coeffs = params['harm']

    with timer.time_with_cm(
            "simulation/sort trajectories/calculate costs/calculate risk/"
            + "collision probability"
    ):

        if modes["fast_prob_mahalanobis"]:
            coll_prob_dict = get_inv_mahalanobis_dist(
                traj=traj,
                predictions=predictions,
                vehicle_params=vehicle_params,
            )

        else:
            coll_prob_dict = get_collision_probability_fast(
                traj=traj,
                predictions=predictions,
                vehicle_params=vehicle_params,
                start_idx=start_idx,
                mode_num=mode_num,
            )

    ego_harm_traj, obst_harm_traj = get_harm(
        scenario, traj, predictions, ego_id, vehicle_params, modes, coeffs, timer, start_idx, mode_num
    )

    # Calculate risk out of harm and collision probability
    ego_risk_max = {}
    ego_harm_max = {}
    obst_risk_max = {}
    obst_harm_max = {}

    for key in ego_harm_traj:
        ego_risk_traj_list = [[None]] * len(ego_harm_traj[key])
        obst_risk_traj_list = [[None]] * len(ego_harm_traj[key])
        # iterate over the modes per obstacle
        for mode in range(len(ego_harm_traj[key])):
            '''
            ego_risk_traj_list[mode] = [
                ego_harm_traj[key][mode][t] * coll_prob_dict[key][mode][t]
                for t in range(len(ego_harm_traj[key][mode]))
            ]
            '''
            ego_risk_traj_list[mode] = [
                coll_prob_dict[key][mode][t]
                for t in range(len(ego_harm_traj[key][mode]))
            ]
            obst_risk_traj_list[mode] = [
                coll_prob_dict[key][mode][t]
                for t in range(len(obst_harm_traj[key][mode]))
            ]

        ego_harm_traj_list = ego_harm_traj[key]
        obst_harm_traj_list = obst_harm_traj[key]

        # For the shared plan, we need to weight the probability of collision of the shared
        # plan based on the beliefs over the modes
        if mode_num == 100:
            belief_idx = 0
            # This means this is the shared part of the plan
            for mode in range(len(ego_harm_traj[key])):

                if mode % 3 == 0:
                    belief_idx += 1
                ego_risk_traj_list[mode] = [belief[belief_idx - 1] * num for num in ego_risk_traj_list[mode]]
                ego_harm_traj_list[mode] = [belief[belief_idx - 1] * num for num in ego_harm_traj_list[mode]]
                obst_harm_traj_list[mode] = [belief[belief_idx - 1] * num for num in obst_harm_traj_list[mode]]
                obst_risk_traj_list[mode] = [belief[belief_idx - 1] * num for num in obst_risk_traj_list[mode]]
                '''
              
                ego_risk_traj_list[mode] = [belief[mode + 3] * num for num in ego_risk_traj_list[mode]]
                ego_harm_traj_list[mode] = [belief[mode + 3] * num for num in ego_harm_traj_list[mode]]
                obst_harm_traj_list[mode] = [belief[mode + 3] * num for num in obst_harm_traj_list[mode]]
                obst_risk_traj_list[mode] = [belief[mode + 3] * num for num in obst_risk_traj_list[mode]]
                '''

        # Take max as representative for the whole trajectory
        ego_risk_max[key] = max(max(row) for row in ego_risk_traj_list)
        ego_harm_max[key] = max(max(row) for row in ego_harm_traj_list)
        obst_harm_max[key] = max(max(row) for row in obst_harm_traj_list)
        obst_risk_max[key] = max(max(row) for row in obst_risk_traj_list)

    # calculate boundary harm
    # col_obj = create_collision_object(traj, vehicle_params, ego_state)
    '''
    leaving_road_at = trajectory_queries.trajectories_collision_static_obstacles(
        trajectories=[col_obj],
        static_obstacles=road_boundary,
        method="grid",
        num_cells=32,
        auto_orientation=True,
    )

    if leaving_road_at[0] != -1:
        coll_time_step = leaving_road_at[0] - ego_state.time_step
        coll_vel = traj.v[coll_time_step]

        boundary_harm = get_protected_inj_prob_log_reg_ignore_angle(
            velocity=coll_vel, coeff=coeffs
        )

    else:
        boundary_harm = 0
    '''
    boundary_harm = 0

    return ego_risk_max, obst_risk_max, ego_harm_max, obst_harm_max, boundary_harm


def get_bayesian_costs(ego_risk_max, obst_risk_max, boundary_harm):
    """
    Bayesian Principle.

    Calculate the risk cost via the Bayesian Principle for the given
    trajectory.

    Args:
        ego_harm_traj (Dict): Dictionary with collision data for all
            obstacles and all time steps.
        obst_harm_traj (Dict): Dictionary with collision data for all
            obstacles and all time steps.

    Returns:
        Dict: Risk costs for the considered trajectory according to the
            Bayesian Principle
    """
    if len(ego_risk_max) == 0:
        return 0

    return (sum(ego_risk_max.values()) + sum(obst_risk_max.values()) + boundary_harm) / (
            len(ego_risk_max) * 2
    )


def get_equality_costs(ego_risk_max, obst_risk_max):
    """
    Equality Principle.

    Calculate the risk cost via the Equality Principle for the given
    trajectory.

    Args:
        ego_harm_traj (Dict): Dictionary with collision data for all
            obstacles and all time steps.
        obst_harm_traj (Dict): Dictionary with collision data for all
            obstacles and all time steps.
        timestep (Int): Currently considered time step.
        equality_mode (Str): Select between normalized ego risk
            ("normalized") and partial risks ("partial").

    Returns:
        float: Risk costs for the considered trajectory according to the
            Equality Principle
    """
    if len(ego_risk_max) == 0:
        return 0

    return sum(
        [abs(ego_risk_max[key] - obst_risk_max[key]) for key in ego_risk_max]
    ) / len(ego_risk_max)


def get_maximin_costs(ego_risk_max, obst_risk_max, ego_harm_max, obst_harm_max, boundary_harm, eps=10e-10,
                      scale_factor=10):
    """
    Maximin Principle.

    Calculate the risk cost via the Maximin principle for the given
    trajectory.

    Args:
        ego_harm_traj (Dict): Dictionary with collision data for all
            obstacles and all time steps.
        obst_harm_traj (Dict): Dictionary with collision data for all
            obstacles and all time steps.
        timestep (Int): Currently considered time step.
        maximin_mode (Str): Select between normalized ego risk
            ("normalized") and partial risks ("partial").

    Returns:
        float: Risk costs for the considered trajectory according to the
            Maximum Principle
    """
    if len(ego_harm_max) == 0:
        return 0

    # Set maximin to 0 if probability (or risk) is 0
    maximin_ego = [a * int(b < eps) for a, b in zip(ego_harm_max.values(), ego_risk_max.values())]
    maximin_obst = [a * int(bool(b < eps)) for a, b in zip(obst_harm_max.values(), obst_risk_max.values())]

    return max(maximin_ego + maximin_obst + [boundary_harm]) ** scale_factor


def get_ego_costs(ego_risk_max, boundary_harm):
    """
    Calculate the utilitarian ego cost for the given trajectory.

    Args:
        ego_harm_traj (Dict): Dictionary with collision data for all
            obstacles and all time steps.
        timestep (Int): Currently considered time step.

    Returns:
        Dict: Utilitarian ego risk cost
    """
    if len(ego_risk_max) == 0:
        return 0

    return sum(ego_risk_max.values()) + boundary_harm


def get_responsibility_cost(scenario, traj, ego_state, obst_risk_max, predictions, reach_set, mode="reach_set"):
    """Get responsibility cost.

    Args:
        obst_risk_max (_type_): _description_
        predictions (_type_): _description_
        mode (str) : "reach set" for reachable set mode, else assignement by space of action

    Returns:
        _type_: _description_

    """
    bool_contain_cache = None
    if "reach_set" in mode and reach_set is not None:
        resp_cost, bool_contain_cache = calc_responsibility_reach_set(traj, ego_state, reach_set)

    else:
        # Assign responsibility to predictions
        predictions = assign_responsibility_by_action_space(
            scenario, ego_state, predictions
        )
        resp_cost = 0

        for key in predictions:
            resp_cost -= predictions[key]["responsibility"] * obst_risk_max[key]

    return resp_cost, bool_contain_cache
