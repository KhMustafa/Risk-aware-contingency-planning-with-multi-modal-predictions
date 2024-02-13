"""Logistic regression harm outer function."""

import numpy as np
from beliefplanning.risk_assessment.helpers.properties import calc_delta_v
from beliefplanning.risk_assessment.utils.logistic_regression_asymmetrical import (
    get_protected_inj_prob_log_reg_complete,
    get_protected_inj_prob_log_reg_reduced,
)
from beliefplanning.risk_assessment.utils.logistic_regression_symmetrical import (
    get_protected_inj_prob_log_reg_complete_sym,
    get_protected_inj_prob_log_reg_ignore_angle,
    get_protected_inj_prob_log_reg_reduced_sym,
)


def get_protected_log_reg_harm(
    ego_vehicle,
    obstacle,
    pdof: float,
    ego_angle: float,
    obs_angle: float,
    modes,
    coeffs,
):
    """
    Select impact area mode for logistic regression models.

    Get the harm for two possible collision partners, if both are vehicles
    with a protective crash structure, via logistic regression.

    Args:
        ego_vehicle (HarmParameters): object with crash relevant ego
            vehicle parameters.
        obstacle (HarmParameters): object with crash relevant obstacle
            parameters.
        pdof (float): Crash angle [rad].
        ego_angle (float): Angle of impact area for the ego vehicle.
        obs_angle (float): Angle of impact area for the obstacle.
        modes (Dict): Risk modes. Read from risk.json.
        coeffs (Dict): Risk parameters. Read from risk_parameters.json.

    Returns:
        float: Harm for ego vehicle
        float: Harm for obstacle vehicle
    """
    # calculate difference between pre-crash and post-crash speed
    ego_delta_v, obstacle_delta_v = calc_delta_v(
        vehicle_1=ego_vehicle, vehicle_2=obstacle, pdof=pdof
    )

    # calculate harm based on angle mode
    if modes["ignore_angle"] is False:
        if modes["sym_angle"] is False:
            if modes["reduced_angle_areas"] is False:
                # use log reg complete
                # calculate harm for the ego vehicle
                ego_harm = get_protected_inj_prob_log_reg_complete(
                    velocity=ego_delta_v, angle=ego_angle, coeff=coeffs
                )

                # calculate harm for the obstacle vehicle
                obstacle_harm = get_protected_inj_prob_log_reg_complete(
                    velocity=obstacle_delta_v, angle=obs_angle, coeff=coeffs
                )

            else:
                # use log reg reduced
                # calculate harm for the ego vehicle
                ego_harm = get_protected_inj_prob_log_reg_reduced(
                    velocity=ego_delta_v, angle=ego_angle, coeff=coeffs
                )

                # calculate harm for the obstacle vehicle
                obstacle_harm = get_protected_inj_prob_log_reg_reduced(
                    velocity=obstacle_delta_v, angle=obs_angle, coeff=coeffs
                )
        else:
            if modes["reduced_angle_areas"] is False:
                # use log reg sym complete
                # calculate harm for the ego vehicle
                ego_harm = get_protected_inj_prob_log_reg_complete_sym(
                    velocity=ego_delta_v, angle=ego_angle, coeff=coeffs
                )

                # calculate harm for the obstacle vehicle
                obstacle_harm = get_protected_inj_prob_log_reg_complete_sym(
                    velocity=obstacle_delta_v, angle=obs_angle, coeff=coeffs
                )
            else:
                # use log reg sym reduced
                # calculate harm for the ego vehicle
                ego_harm = get_protected_inj_prob_log_reg_reduced_sym(
                    velocity=ego_delta_v, angle=ego_angle, coeff=coeffs
                )

                # calculate harm for the obstacle vehicle
                obstacle_harm = get_protected_inj_prob_log_reg_reduced_sym(
                    velocity=obstacle_delta_v, angle=obs_angle, coeff=coeffs
                )
    else:
        # use log reg delta v
        # calculate harm for the ego vehicle
        ego_harm = get_protected_inj_prob_log_reg_ignore_angle(
            velocity=ego_delta_v, coeff=coeffs
        )

        # calculate harm for the obstacle vehicle
        obstacle_harm = get_protected_inj_prob_log_reg_ignore_angle(
            velocity=obstacle_delta_v, coeff=coeffs
        )

    return ego_harm, obstacle_harm


def get_unprotected_log_reg_harm(ego_vehicle, obstacle, pdof, coeff):
    """
    Logistic regression for pedestrians.

    Get the harm for two possible collision partners, if the obstacle is not
    protected by crash structure.

    Args:
        ego_vehicle (HarmParameters): object with crash relevant ego
            vehicle parameters.
        obstacle (HarmParameters): object with crash relevant obstacle
            parameters.
        pdof (float): Crash angle [rad].
        coeff (Dict): Risk parameters. Read from risk_parameters.json.

    Returns:
        float: Harm for ego vehicle
        float: Harm for obstacle
    """
    # calculate difference between pre-crash and post-crash velocity
    ego_delta_v, obstacle_delta_v = calc_delta_v(
        vehicle_1=ego_vehicle, vehicle_2=obstacle, pdof=pdof
    )

    # calc ego harm
    ego_harm = get_protected_inj_prob_log_reg_ignore_angle(
        velocity=ego_delta_v, coeff=coeff
    )

    # calculate obstacle harm
    # logistic regression model
    obstacle_harm = 1 / (
        1
        + np.exp(
            coeff["pedestrian"]["const"]
            - coeff["pedestrian"]["speed"] * obstacle_delta_v
        )
    )

    return ego_harm, obstacle_harm
