#!/user/bin/env python

"""Sampling-based trajectory planning in a frenet frame considering ethical implications."""

# Standard imports
import os
import sys
import copy
import warnings
import json
from inspect import currentframe, getframeinfo
import pathlib
import pickle
import time
import math

# Third party imports
import numpy as np
import matplotlib
from commonroad.planning.planning_problem import PlanningProblem
from commonroad.scenario.scenario import Scenario
from commonroad.scenario.trajectory import State
from commonroad_dc.collision.collision_detection.pycrcc_collision_dispatch import (
    create_collision_checker,
)

from commonroad_helper_functions.exceptions import (
    ExecutionTimeoutError,
)
from prediction import WaleNet
from Init_MPC import initBranchMPC
from MPC_branch import BranchMPC
from highway_branch_dyn import *
import Highway_env_branch

from utils_baseline import Branch_constants

# Custom imports
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

mopl_path = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
sys.path.append(mopl_path)

from beliefplanning.planner.planning import Planner
from beliefplanning.planner.utils.timeout import Timeout
from beliefplanning.planner.Frenet.utils.visualization import draw_frenet_trajectories, \
    draw_contingent_trajectories, draw_all_contingent_trajectories

from beliefplanning.planner.Frenet.utils.prediction_helpers import (
    add_static_obstacle_to_prediction,
    get_dyn_and_stat_obstacles,
    get_ground_truth_prediction,
    get_obstacles_in_radius,
    get_orientation_velocity_and_shape_of_prediction,
    belief_updater,
    get_obstacles_prediction_overtake,
    get_prediction_from_scenario_tree,
)
from beliefplanning.planner.Frenet.configs.load_json import (
    load_harm_parameter_json,
    load_planning_json,
    load_risk_json,
    load_weight_json,
    load_contingency_json,
)
from beliefplanning.planner.Frenet.utils.frenet_functions import (
    calc_frenet_trajectories,
    calc_contingent_plans,
    get_v_list,
    sort_frenet_trajectories,
)
from beliefplanning.planner.Frenet.utils.logging import FrenetLogging
from beliefplanning.planner.utils import reachable_set
from beliefplanning.risk_assessment.visualization.risk_visualization import (
    create_risk_files,
)
from beliefplanning.risk_assessment.visualization.risk_dashboard import risk_dashboard


class FrenetPlanner(Planner):
    """Jerk optimal planning in frenet coordinates with quintic polynomials in lateral direction and quartic
    polynomials in longitudinal direction."""

    def __init__(
            self,
            scenario: Scenario,
            planning_problem: PlanningProblem,
            ego_id: int,
            vehicle_params,
            mode,
            exec_timer=None,
            frenet_parameters: dict = None,
            contingency_parameters: dict = None,
            sensor_radius: float = 55.0,
            plot_frenet_trajectories: bool = False,
            weights=None,
            settings=None,
    ):
        """
        Initialize a frenét planner.

        Args:
            scenario (Scenario): Scenario.
            planning_problem (PlanningProblem): Given planning problem.
            ego_id (int): ID of the ego vehicle.
            vehicle_params (VehicleParameters): Parameters of the ego vehicle.
            mode (Str): Mode of the frenét planner.
            timing (bool): True if the execution times should be saved. Defaults to False.
            frenet_parameters (dict): Parameters for the frenét planner. Defaults to None.
            sensor_radius (float): Radius of the sensor model. Defaults to 30.0.
            plot_frenet_trajectories (bool): True if the frenét paths should be visualized. Defaults to False.
            weights(dict): the weights of the costfunction. Defaults to None.
        """
        super().__init__(scenario, planning_problem, ego_id, vehicle_params, exec_timer)
        self.N_lane = None
        self.mpc = None

        self.traj_rec = []
        self.state_rec = []
        self.zPred_rec = []
        self.exec_time = []
        self.branch_w_rec = []

        self.long_jerk = []
        self.lat_jerk = []

        # Set up logger
        self.logger = FrenetLogging(
            log_path=f"./planner/Frenet/results/logs/{scenario.benchmark_id}.csv"
        )

        try:
            with Timeout(10, "Frenet Planner initialization"):

                self.exec_timer.start_timer("initialization/total")
                if frenet_parameters is None:
                    print(
                        "No frenet parameters found. Swichting to default parameters."
                    )
                    frenet_parameters = {
                        "t_list": [2.0],
                        "v_list_generation_mode": "linspace",
                        "n_v_samples": 5,
                        "d_list": np.linspace(-3.5, 3.5, 15),
                        "dt": 0.1,
                        "v_thr": 3.0,
                    }

                # parameters for frenet planner
                self.frenet_parameters = frenet_parameters

                # parameters for contingency planner
                self.contingency_parameters = contingency_parameters

                # vehicle parameters
                self.p = vehicle_params

                # load parameters
                self.params_harm = load_harm_parameter_json()
                if weights is None:
                    self.params_weights = load_weight_json()
                else:
                    self.params_weights = weights
                if settings is not None:
                    if "risk_dict" in settings:
                        self.params_mode = settings["risk_dict"]
                    else:
                        self.params_mode = load_risk_json()

                self.params_dict = {
                    'weights': self.params_weights,
                    'modes': self.params_mode,
                    'harm': self.params_harm,
                }

                self.v_goal_min = None
                self.v_goal_max = None

                self.cost_dict = {}

                # check if the planning problem has an initial acceleration, else set it to zero
                if not hasattr(self.ego_state, "acceleration"):
                    self.ego_state.acceleration = 0.0

                # initialize the driven trajectory with the initial position
                self.driven_traj = [
                    State(
                        position=self.ego_state.position,
                        orientation=self.ego_state.orientation,
                        time_step=self.ego_state.time_step,
                        velocity=self.ego_state.velocity,
                        acceleration=self.ego_state.acceleration,
                    )
                ]

                # get sensor radius param, and planner mode
                self.sensor_radius = sensor_radius
                self.mode = mode

                # get visualization marker
                self.plot_frenet_trajectories = plot_frenet_trajectories

                # initialize the prediction network if necessary
                if self.mode == "WaleNet" or self.mode == "risk":

                    prediction_config_path = os.path.join(
                        os.path.dirname(os.path.abspath(__file__)),
                        "configs",
                        "prediction.json",
                    )
                    with open(prediction_config_path, "r") as f:
                        online_args = json.load(f)

                    self.predictor = WaleNet(scenario=scenario, online_args=online_args, verbose=False)
                elif self.mode == "ground_truth":
                    self.predictor = None
                else:
                    raise ValueError("mode must be ground_truth, WaleNet, or risk")

                # check whether reachable sets have to be calculated for responsibility
                if (
                        'responsibility' in self.params_weights
                        and self.params_weights['responsibility'] > 0
                ):
                    self.responsibility = True
                    self.reach_set = reachable_set.ReachSet(
                        scenario=self.scenario,
                        ego_id=self.ego_id,
                        ego_length=self.p.l,
                        ego_width=self.p.w,
                    )
                else:
                    self.responsibility = False
                    self.reach_set = None

                with self.exec_timer.time_with_cm(
                        "initialization/initialize collision checker"
                ):
                    cc_scenario = copy.deepcopy(self.scenario)
                    cc_scenario.remove_obstacle(
                        obstacle=[cc_scenario.obstacle_by_id(ego_id)]
                    )
                    try:
                        self.collision_checker = create_collision_checker(cc_scenario)
                    except Exception:
                        raise BrokenPipeError("Collision Checker fails.") from None
                self.exec_timer.stop_timer("initialization/total")
        except ExecutionTimeoutError:
            raise TimeoutError

    def sim_overtake(self):
        N = 8  # number of time steps for each branch
        n = 4
        d = 2  # State and Input dimension
        x0 = np.array(
            [0, 1.8, 0,
             0])  # Initial condition (only for initializing the MPC, not the actual initial state of the sim)
        am = 6.0
        rm = 0.3
        dt = 0.1
        NB = 2  # number of branching, 2 means a tree with 1-m-m^2 branches at each level.

        N_lane = 4

        # Initialize controller parameters
        xRef = np.array([0.5, 1.8, 15, 0])
        cons = Branch_constants(s1=2, s2=3, c2=0.5, tran_diag=0.3, alpha=1, R=1.2, am=am, rm=rm, J_c=20, s_c=1, ylb=0.,
                                yub=7.2, L=4, W=2.5, col_alpha=5, Kpsi=0.1)
        backupcons = [lambda x: backup_maintain(x, cons), lambda x: backup_brake(x, cons), lambda x: backup_lc(x, xRef)]
        model = PredictiveModel(n, d, N, backupcons, dt, cons)

        mpcParam = initBranchMPC(n, d, N, NB, xRef, am, rm, N_lane, cons.W)
        mpc = BranchMPC(mpcParam, model)

        backup, zPred, self.obst_new_state, branch_w, state_rec = Highway_env_branch.sim_overtake(mpc, N_lane,
                                                                                                  self.time_step,
                                                                                                  self.ego_state,
                                                                                                  self.obst_new_state,
                                                                                                  self._trajectory)
        self.N_lane = N_lane
        self.mpc = mpc
        return backup, zPred, state_rec, branch_w

    def _step_planner(self):
        """Frenet Planner step function.

        These methods overload the basic step method. It generates a new trajectory with the jerk optimal polynomials.
        """
        self.exec_timer.start_timer("simulation/total")

        with self.exec_timer.time_with_cm("simulation/update driven trajectory"):
            # update the driven trajectory
            # add the current state to the driven path
            if self.ego_state.time_step != 0:
                current_state = State(
                    position=self.ego_state.position,
                    orientation=self.ego_state.orientation,
                    time_step=self.ego_state.time_step,
                    velocity=self.ego_state.velocity,
                    acceleration=self.ego_state.acceleration,
                )

                self.driven_traj.append(current_state)

        # find position along the reference spline (s, s_d, s_dd, d, d_d, d_dd)
        c_s = self.trajectory["s_loc_m"][1]
        c_s_d = self.ego_state.velocity
        c_s_dd = self.ego_state.acceleration
        if self.time_step == 0:
            # c_d = -3.6
            c_d = 0
        else:
            c_d = self.trajectory["d_loc_m"][1]

        c_d_d = self.trajectory["d_d_loc_mps"][1]
        c_d_dd = self.trajectory["d_dd_loc_mps2"][1]

        # get the end velocities for the frenét paths
        current_v = self.ego_state.velocity
        max_acceleration = self.p.longitudinal.a_max
        t_min = min(self.frenet_parameters["t_list"])
        t_max = max(self.frenet_parameters["t_list"])
        max_v = min(
            current_v + (max_acceleration / 2.0) * t_max, self.p.longitudinal.v_max
        )
        min_v = max(0.01, current_v - max_acceleration * t_min)

        with self.exec_timer.time_with_cm("simulation/get v list"):
            v_list = get_v_list(
                v_min=min_v,
                v_max=max_v,
                v_cur=current_v,
                v_goal_min=self.v_goal_min,
                v_goal_max=self.v_goal_max,
                mode=self.frenet_parameters["v_list_generation_mode"],
                n_samples=self.frenet_parameters["n_v_samples"],
            )

        with self.exec_timer.time_with_cm("simulation/calculate trajectories/total"):
            # d_list = self.frenet_parameters["d_list"]
            d_list = np.linspace(-3.6, 0, 5)
            t_list = self.frenet_parameters["t_list"]

            # if self.ego_state.time_step == 0 or self.open_loop is False:
            ft_list = calc_frenet_trajectories(
                c_s=c_s,
                c_s_d=c_s_d,
                c_s_dd=c_s_dd,
                c_d=c_d,
                c_d_d=c_d_d,
                c_d_dd=c_d_dd,
                d_list=d_list,
                t_list=t_list,
                v_list=v_list,
                dt=self.frenet_parameters["dt"],
                csp=self.reference_spline,
                v_thr=self.frenet_parameters["v_thr"],
                exec_timer=self.exec_timer,
                t_min=t_min,
                t_max=t_max,
                max_acceleration=max_acceleration,
                max_velocity=self.p.longitudinal.v_max,
                v_goal_min=self.v_goal_min,
                v_goal_max=self.v_goal_max,
                mode=self.frenet_parameters["v_list_generation_mode"],
                n_samples=self.frenet_parameters["n_v_samples"],
                contin=False
            )

        # we need to get the prediction from
        with self.exec_timer.time_with_cm("simulation/prediction"):
            # Overwrite later
            visible_area = None
            backup, zPred, state_rec, branch_w = self.sim_overtake()
            # predictions = get_obstacles_prediction_overtake(zPred, backup)
            predictions = get_prediction_from_scenario_tree(zPred)
            self.scenario.dynamic_obstacles[0].initial_state.position[0] = state_rec[1][0][0]
            self.scenario.dynamic_obstacles[0].initial_state.position[1] = -state_rec[1][0][1]
            self.scenario.dynamic_obstacles[0].initial_state.orientation = state_rec[1][0][3]

        # calculate reachable sets
        if self.responsibility:
            with self.exec_timer.time_with_cm(
                    "simulation/calculate and check reachable sets"
            ):
                self.reach_set.calc_reach_sets(self.ego_state, list(predictions.keys()))

        with (self.exec_timer.time_with_cm("simulation/sort trajectories/total")):
            # sorted list (increasing costs)

            # if self.ego_state.time_step == 0 or self.open_loop is False:
            # belief = [self.belief[self.ego_state.time_step], 1 - self.belief[self.ego_state.time_step]]
            belief = branch_w
            # belief = [1] * 12
            ft_list_valid, ft_list_invalid, validity_dict = sort_frenet_trajectories(
                ego_state=self.ego_state,
                fp_list=ft_list,
                global_path=self.global_path,
                predictions=predictions,
                mode=self.mode,
                params=self.params_dict,
                planning_problem=self.planning_problem,
                scenario=self.scenario,
                vehicle_params=self.p,
                ego_id=self.ego_id,
                dt=self.frenet_parameters["dt"],
                sensor_radius=self.sensor_radius,
                collision_checker=self.collision_checker,
                exec_timer=self.exec_timer,
                start_idx=0,
                mode_num=100,
                belief=belief,
                reach_set=(self.reach_set if self.responsibility else None)
            )

            with self.exec_timer.time_with_cm(
                    "simulation/sort trajectories/sort list by costs"
            ):
                # Sort the list of frenet trajectories (minimum cost first):
                ft_list_valid.sort(key=lambda fp: fp.cost, reverse=False)

                max_acceleration = self.p.longitudinal.a_max
                t_min = min(self.contingency_parameters["t_list"])
                t_max = max(self.contingency_parameters["t_list"])

                # d_list = self.contingency_parameters["d_list"]
                d_list = np.linspace(-3.6, 0, 3)
                t_list = self.contingency_parameters["t_list"]

                ft_final_list = []

                for plan in ft_list_valid:
                    final_plan = {}

                    max_v = min(
                        plan.v[-1] + (max_acceleration / 2.0) * t_max, self.p.longitudinal.v_max
                    )
                    min_v = max(0.01, plan.v[-1] - max_acceleration * t_min)

                    # Plan contingent plans for only valid shared trajectories
                    v_list = get_v_list(
                        v_min=min_v,
                        v_max=max_v,
                        v_cur=plan.v[-1],
                        v_goal_min=self.v_goal_min,
                        v_goal_max=self.v_goal_max,
                        mode=self.contingency_parameters["v_list_generation_mode"],
                        n_samples=self.contingency_parameters["n_v_samples"],
                    )

                    final_plan['shared_plan'] = plan

                    if t_list[0] != 0:
                        # only calculate contingent plans when needed
                        ft_contingent_list = calc_frenet_trajectories(
                            c_s=plan.s[-1],
                            c_s_d=plan.s_d[-1],
                            c_s_dd=plan.s_dd[-1],
                            c_d=plan.d[-1],
                            c_d_d=plan.d_d[-1],
                            c_d_dd=plan.d_dd[-1],
                            d_list=d_list,
                            t_list=t_list,
                            v_list=v_list,
                            dt=self.contingency_parameters["dt"],
                            csp=self.reference_spline,
                            v_thr=self.contingency_parameters["v_thr"],
                            exec_timer=self.exec_timer,
                            t_min=t_min,
                            t_max=t_max,
                            max_acceleration=max_acceleration,
                            max_velocity=self.p.longitudinal.v_max,
                            v_goal_min=self.v_goal_min,
                            v_goal_max=self.v_goal_max,
                            mode=self.contingency_parameters["v_list_generation_mode"],
                            n_samples=self.contingency_parameters["n_v_samples"],
                            contin=True
                        )

                        # we want to calculate the best contingent plan per mode
                        for mode_num in range(len(predictions[1]['pos_list'])):
                            ft_conting_list_valid, ft_conting_list_invalid, validity_conting_dict = sort_frenet_trajectories(
                                ego_state=self.ego_state,
                                fp_list=ft_contingent_list,
                                global_path=self.global_path,
                                predictions=predictions,
                                mode=self.mode,
                                params=self.params_dict,
                                planning_problem=self.planning_problem,
                                scenario=self.scenario,
                                vehicle_params=self.p,
                                ego_id=self.ego_id,
                                dt=self.frenet_parameters["dt"],
                                sensor_radius=self.sensor_radius,
                                collision_checker=self.collision_checker,
                                exec_timer=self.exec_timer,
                                start_idx=int(max(self.frenet_parameters["t_list"]) / self.frenet_parameters["dt"]),
                                mode_num=mode_num,
                                reach_set=(self.reach_set if self.responsibility else None)
                            )
                            # Sort the list of contingent trajectories (minimum cost first):
                            ft_conting_list_valid.sort(key=lambda fp: fp.cost, reverse=False)
                            final_plan[mode_num] = ft_conting_list_valid[0]

                    ft_final_list.append(final_plan)

                # we need to get the belief over the modes to use it as weights in the cost function
                '''
                self.belief = belief_updater(predictions, self.belief)
                self.belief_list.append(self.belief[0])
                '''
                # iterate over the final frenet list, and assign a cost to the entire traj
                # print('belief is: ', belief[0])
                for plan in ft_final_list:
                    if len(plan) == 1:
                        # This means we have only a single plan along the horizon
                        plan['cost'] = plan['shared_plan'].cost
                    else:
                        plan['cost'] = plan['shared_plan'].cost + branch_w[0] * plan[0].cost + branch_w[1] * plan[
                            3].cost
                        + branch_w[2] * plan[6].cost

                # sort the final plan
                ft_final_list.sort(key=lambda fp: fp['cost'], reverse=False)

        with self.exec_timer.time_with_cm("plot trajectories"):
            # if self.ego_state.time_step == 0 or self.open_loop == False:
            if self.params_mode["figures"]["create_figures"] is True:
                if self.mode == "risk":
                    create_risk_files(
                        scenario=self.scenario,
                        time_step=self.ego_state.time_step,
                        destination=os.path.join(os.path.dirname(__file__), "results"),
                        risk_modes=self.params_mode,
                        weights=self.params_weights,
                        marked_vehicle=self.ego_id,
                        planning_problem=self.planning_problem,
                        traj=ft_list_valid,
                        global_path=self.global_path_to_goal,
                        global_path_after_goal=self.global_path_after_goal,
                        driven_traj=self.driven_traj,
                    )

                else:
                    warnings.warn(
                        "Harm diagrams could not be created."
                        "Please select mode risk.",
                        UserWarning,
                    )

            if self.params_mode["risk_dashboard"] is True:
                if self.mode == "risk":
                    risk_dashboard(
                        scenario=self.scenario,
                        time_step=self.ego_state.time_step,
                        destination=os.path.join(
                            os.path.dirname(__file__), "results/risk_plots"
                        ),
                        risk_modes=self.params_mode,
                        weights=self.params_weights,
                        planning_problem=self.planning_problem,
                        traj=(ft_list_valid + ft_list_invalid),
                    )

                else:
                    warnings.warn(
                        "Risk dashboard could not be created."
                        "Please select mode risk.",
                        UserWarning,
                    )

            # print some information about the frenet trajectories
            if self.plot_frenet_trajectories:
                matplotlib.use("TKAgg")
                print(
                    "Time step: {} | Velocity: {:.2f} m/s | Acceleration: {:.2f} m/s2".format(
                        self.time_step, current_v, c_s_dd
                    )
                )
                '''
                Highway_env_branch.plot_scenario(self.mpc, self.N_lane, self.time_step, self.ego_state,
                                                 self.obst_new_state, ft_final_list[0],
                                                 state_rec, zPred)
                '''
                self.traj_rec.append(ft_final_list[0])
                self.state_rec.append(state_rec)
                self.zPred_rec.append(zPred)
                self.branch_w_rec.append(branch_w)

                if self.time_step == 100:
                    Highway_env_branch.plot_scenario(self.mpc, self.N_lane, self.time_step, self.ego_state,
                                                     self.obst_new_state, self.traj_rec,
                                                     self.state_rec, self.zPred_rec)

            try:

                draw_all_contingent_trajectories(
                    scenario=self.scenario,
                    time_step=self.ego_state.time_step,
                    marked_vehicle=self.ego_id,
                    planning_problem=self.planning_problem,
                    traj=None,
                    global_path=self.global_path_to_goal,
                    global_path_after_goal=self.global_path_after_goal,
                    driven_traj=self.driven_traj,
                    animation_area=50.0,
                    predictions=predictions,
                    visible_area=visible_area,
                    valid_traj=ft_final_list,
                    best_traj=self.contingency_trajectory,
                    open_loop=self.open_loop,
                )

            except Exception as e:
                print(e)

            if self.ego_state.time_step == 0:
                self.contingency_trajectory = ft_final_list

            # best trajectory
            if len(ft_list_valid) > 0:
                best_trajectory = ft_list_valid[0]
            elif len(ft_list_invalid) > 0:
                best_trajectory = ft_list_invalid[0]
                # raise NoLocalTrajectoryFoundError('Failed. No valid frenét path found')

        self.exec_timer.stop_timer("simulation/total")

        # this should work fine, we don't need the entire trajectory since only the 2nd index is required
        # to update the initial state.
        self._trajectory = {
            "s_loc_m": best_trajectory.s,
            "d_loc_m": best_trajectory.d,
            "d_d_loc_mps": best_trajectory.d_d,
            "d_dd_loc_mps2": best_trajectory.d_dd,
            "x_m": best_trajectory.x,
            "y_m": best_trajectory.y,
            "psi_rad": best_trajectory.yaw,
            "kappa_radpm": best_trajectory.curv,
            "v_mps": best_trajectory.s_d,
            "ax_mps2": best_trajectory.s_dd,
            "time_s": best_trajectory.t,
        }
        return ft_final_list[0], state_rec, zPred, self.obst_new_state


if __name__ == "__main__":
    import argparse
    from planner.plannertools.evaluate import ScenarioEvaluator
    from planner.Frenet.plannertools.frenetcreator import FrenetCreator

    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="recorded/hand-crafted/DEU_Muc-4_2_T-1"
                                              ".xml")
    parser.add_argument("--time", action="store_true")
    args = parser.parse_args()

    if "commonroad" in args.scenario:
        scenario_path = args.scenario.split("scenarios/")[-1]
    else:
        scenario_path = args.scenario

    # load settings from planning_fast.json
    settings_dict = load_planning_json("planning_fast.json")
    settings_dict["contingency_settings"] = load_contingency_json("contingency.json")
    settings_dict["risk_dict"] = risk_dict = load_risk_json()
    if not args.time:
        settings_dict["evaluation_settings"]["show_visualization"] = True
    eval_directory = (
        pathlib.Path(__file__).resolve().parents[0].joinpath("results").joinpath("eval")
    )
    # Create the frenet creator
    frenet_creator = FrenetCreator(settings_dict)

    # Create the scenario evaluator
    evaluator = ScenarioEvaluator(
        planner_creator=frenet_creator,
        vehicle_type=settings_dict["evaluation_settings"]["vehicle_type"],
        path_to_scenarios=pathlib.Path(
            os.path.join(mopl_path, "beliefplanning/scenarios/")
        ).resolve(),
        log_path=pathlib.Path("./log/example").resolve(),
        collision_report_path=eval_directory,
        timing_enabled=settings_dict["evaluation_settings"]["timing_enabled"],
    )


    def main():
        """Loop for cProfile."""
        _ = evaluator.eval_scenario(scenario_path)


    if args.time:
        import cProfile

        cProfile.run('main()', "output.dat")
        no_trajectores = settings_dict["frenet_settings"]["frenet_parameters"]["n_v_samples"] * len(
            settings_dict["frenet_settings"]["frenet_parameters"]["d_list"])
        import pstats

        sortby = pstats.SortKey.CUMULATIVE
        with open(f"cProfile/{scenario_path.split('/')[-1]}_{no_trajectores}.txt", "w") as f:
            p = pstats.Stats("output.dat", stream=f).sort_stats(sortby)
            p.sort_stats(sortby).print_stats()
    else:
        main()

# EOF
