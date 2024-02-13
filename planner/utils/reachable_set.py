"""Simple reachable sets for objects in scenario."""
import os
import json
import numpy as np

from commonroad.scenario.scenario import Scenario
from beliefplanning.planner.GlobalPath.lanelet_based_planner import (
    find_lanelet_by_position_and_orientation,
)
from beliefplanning.planner.utils import reachable_set_simple
from beliefplanning.planner.utils.reachable_set_simple import simple_reachable_set
from beliefplanning.planner.utils.responsibility import polygon_padding
import pygeos


class ReachSet(object):
    """
    Wrapper for simple reachable sets.

    Wrapper for simple reachable sets of
    all dynamic obstacles in scenario except ego.
    Currently the only supported obstacle type is car.

    """

    def __init__(
        self,
        scenario: Scenario,
        ego_id: int,
        ego_length: float,
        ego_width: float,
    ) -> None:
        """
        Initialize reachable set wrapper.

        Args:
        scenario (Scenario): Scenario.
        ego_id (int): ID of the ego vehicle.
        ego_length (float): length of ego vehicle.
        ego_width (float): width of ego vehicle.
        """
        # reach_set_params (dict): Dictionary containing reachable set params:
        # dt (float): desired temporal resolution of the reachable set
        # t_max (float): maximum temporal horizon for the reachable set
        # a_max (float): assumed maximum acceleration of obstacle
        # rules (dict): Dictionary containing rules and their parameters
        self.reach_set_params = load_reach_set_json()
        if self.reach_set_params is None:
            self.reach_set_params = {
                "dt": 0.2,
                "t_max": 2,
                "a_max": 8,
                "depth": 3,
                "rules": {"safe_distance": {"safe_distance_frac": 1.0}},
            }

        self.scenario = scenario
        self.ego_id = ego_id
        self.ego_length = ego_length
        self.ego_width = ego_width

        # simple reachable sets of all obstacles in scenario except ego
        self.reach_sets = {}

        # dictionary containing ReachSetSimple objects for all
        # lanelets that any obstacle has been on
        # key: lanelet id (int)
        self.reach_set_objs = {}

        # dictionary containing ReachSetSimple objects for
        # lanelets, ignoring laterally adjacent lanelets
        # key: lanelet id (int)
        self.reach_set_objs_single = {}

        # reachable set of ego (extended to account for safe distance)
        # does not consider laterally adjacent lanelets
        self.ego_reach_set = {}

    def calc_reach_sets(self, ego_state, obstacle_list=None):
        """
        Calculate reachable sets.

        Calculate reachable sets of all dynamic obstacles in scenario except ego.

        Args:
        ego_state (Commonroad State object): state of ego vehicle at time_step
        """
        if "safe_distance" in self.reach_set_params["rules"]:
            self._ego_reach_set(ego_state)

        if obstacle_list is not None:
            obstacles = [self.scenario.obstacle_by_id(obst_id) for obst_id in obstacle_list]
        else:
            obstacles = self.scenario.obstacles

        self.reach_sets[ego_state.time_step] = {}
        # calculate polygon array for self._reach_set_difference(), avoid repeating
        b_dict = {}
        for step in self.ego_reach_set[ego_state.time_step][0]:
            b_poly_pygeos = [b_set[step] for b_set in self.ego_reach_set[ego_state.time_step]]
            len_max = max(len(b_set[step]) for b_set in self.ego_reach_set[ego_state.time_step])
            b_poly_pygeos = polygon_padding(len_max, b_poly_pygeos)
            b_poly_pygeos = pygeos.polygons(b_poly_pygeos)
            # b_poly_pygeos_test = [pygeos.polygons(b_set[step]) for b_set in self.ego_reach_set[ego_state.time_step]]

            b_poly_pygeos = pygeos.union_all(b_poly_pygeos)

            # b[step] = b_poly
            b_dict[step] = b_poly_pygeos

        # uncomment to log ego
        # self.reach_sets[ego_state.time_step][self.ego_id] = self.ego_reach_set[ego_state.time_step]
        for obstacle in obstacles:
            o_id = obstacle.obstacle_id
            if o_id != self.ego_id:
                # get all lanelet ids of obstacle
                l_ids = find_lanelet_by_position_and_orientation(
                    self.scenario.lanelet_network,
                    obstacle.prediction.trajectory.state_list[ego_state.time_step].position,
                    obstacle.prediction.trajectory.state_list[ego_state.time_step].orientation,
                )
                all_ids = [int(i) for i in self.reach_set_objs.keys()]
                new_ids = set(l_ids) - set(all_ids)

                for l_id in new_ids:
                    all_ids = [int(i) for i in self.reach_set_objs.keys()]
                    if l_id not in all_ids:
                        # if new lanelet, create new ReachSetSimple objects
                        (parallel_lanelets, _, _) = self._get_parallel_lanelets(l_id)
                        bounds = self._calc_bounds_rec(
                            lanelet_id=l_id,
                            depth=self.reach_set_params["parameters"]["depth"],
                        )
                        for lnlet_id in parallel_lanelets:
                            self.reach_set_objs[lnlet_id] = []

                        # create a new object for each boundry
                        for (l, r) in bounds:
                            # reach set object trimmed to lanelet bounds
                            obj = reachable_set_simple.ReachSetSimple(
                                bound_l=l, bound_r=r
                            )
                            # same bounds for laterally adjacent lanes in same direction
                            for lnlet_id in parallel_lanelets:
                                self.reach_set_objs[lnlet_id].append(obj)

                self.reach_sets[ego_state.time_step][o_id] = []

                # call simple_reachable_set() before for loop, avoid unnecessary repeating
                srs = simple_reachable_set(
                    obj_pos=obstacle.prediction.trajectory.state_list[
                        ego_state.time_step
                    ].position,
                    obj_heading=obstacle.prediction.trajectory.state_list[
                        ego_state.time_step
                    ].orientation,
                    obj_vel=obstacle.prediction.trajectory.state_list[
                        ego_state.time_step
                    ].velocity,
                    obj_length=obstacle.obstacle_shape.length,
                    obj_width=obstacle.obstacle_shape.width,
                    dt=self.reach_set_params["parameters"]["dt"],
                    t_max=self.reach_set_params["parameters"]["t_max"],
                    a_max=self.reach_set_params["parameters"]["a_max"]
                )
                srs_t = pygeos.polygons([srs[t_key] for t_key in srs.keys()])

                for l_id in l_ids:
                    for reach_set_obj in self.reach_set_objs[l_id]:
                        # adjust call of calc_reach_set()
                        # rs = reach_set_obj.calc_reach_set(
                        #     obj_pos=obstacle.prediction.trajectory.state_list[
                        #         ego_state.time_step
                        #     ].position,
                        #     obj_heading=obstacle.prediction.trajectory.state_list[
                        #         ego_state.time_step
                        #     ].orientation,
                        #     obj_vel=obstacle.prediction.trajectory.state_list[
                        #         ego_state.time_step
                        #     ].velocity,
                        #     obj_length=obstacle.obstacle_shape.length,
                        #     obj_width=obstacle.obstacle_shape.width,
                        #     dt=self.reach_set_params["parameters"]["dt"],
                        #     t_max=self.reach_set_params["parameters"]["t_max"],
                        #     a_max=self.reach_set_params["parameters"]["a_max"],
                        # )
                        rs = reach_set_obj.calc_reach_set(srs, srs_t)

                        if "safe_distance" in self.reach_set_params["rules"]:

                            # adjust call of self._reach_set_difference()
                            # subtract safe distance polygon
                            # reach_set_diffs = self._reach_set_difference(
                            #     rs, self.ego_reach_set[ego_state.time_step]
                            # )
                            reach_set_diffs = self._reach_set_difference(
                                rs, b_dict)

                            self.reach_sets[ego_state.time_step][o_id] += reach_set_diffs
                        else:
                            self.reach_sets[ego_state.time_step][o_id].append(rs)

    def _calc_bounds_rec(self, lanelet_id, depth, lateral=True):
        """
        Bounds considering current and possible successor lanelets.

        Bounds of lanelet with id lanelet_id and all possible successor
        lanelets until maximum depth is reached.

        Args:
        lanelet_id (int): id of starting lanelet.
        depth (int): maximum depth of considered successor lanelets.
        lateral (bool): true indicates that laterally adjacent lanes are considered.

        Returns:
        list((np.ndarray, np.ndarray)): List of boundaries of possible lanes, which are
        tuples of left and right boundaries.
        """
        if depth < 0:
            return []

        bound_list = []
        if not lateral:
            lnlet = self.scenario.lanelet_network.find_lanelet_by_id(lanelet_id)
            successors = lnlet.successor
            bound_l = lnlet.left_vertices
            bound_r = lnlet.right_vertices
        else:
            # get lanelet bounds
            (lanelets, bound_l, bound_r) = self._get_parallel_lanelets(lanelet_id)

            # non-parallel successor lanelets
            successors = set()
            for lnlet in lanelets:
                suc = self.scenario.lanelet_network.find_lanelet_by_id(lnlet).successor
                if suc is not None:
                    successors = successors.union(set(suc))
            successors = self._get_non_parallel_lanelets(successors)

        if depth == 0 or len(successors) == 0:
            bound_list.append((bound_l, bound_r))
            return bound_list

        # append bounds of successor lanelets
        for successor in successors:
            bounds = self._calc_bounds_rec(successor, depth - 1, lateral)
            for (l, r) in bounds:
                bound_list.append((np.append(bound_l, l, 0), np.append(bound_r, r, 0)))

        return bound_list

    def _get_parallel_lanelets(self, lanelet_id):
        """
        Get all laterally adjacent lanelets in same direction.

        Returns:
        List of ids of laterally adjacent lanelets in same direction.
        Outmost left boundary of all adjacent lanelets.
        Outmost right boundary of all adjacent lanelets.
        """
        adj_left = []
        adj_right = []
        curr = lanelet_id
        left_most = curr
        # find leftmost lanelet in same direction
        while self.scenario.lanelet_network.find_lanelet_by_id(
            curr
        )._adj_left_same_direction:
            lnlet = self.scenario.lanelet_network.find_lanelet_by_id(curr)
            curr = lnlet._adj_left
            left_most = curr
            adj_left.append(curr)

        curr = lanelet_id
        right_most = curr
        # find rightmost lanelet in same direction
        while self.scenario.lanelet_network.find_lanelet_by_id(
            curr
        )._adj_right_same_direction:
            lnlet = self.scenario.lanelet_network.find_lanelet_by_id(curr)
            curr = lnlet._adj_right
            right_most = curr
            adj_right.append(curr)

        parallels = adj_left + [lanelet_id] + adj_right
        lnlet = self.scenario.lanelet_network.find_lanelet_by_id(left_most)
        bound_l = lnlet.left_vertices
        lnlet = self.scenario.lanelet_network.find_lanelet_by_id(right_most)
        bound_r = lnlet.right_vertices
        return parallels, bound_l, bound_r

    def _get_non_parallel_lanelets(self, lanelets):
        """
        Get lanelets which aren't laterally adjacent and in same direction.

        Get lanelets such that they are pairwise
        not laterally adjacent or not in the same direction.

        Returns:
        List of lanelet ids.
        """
        final = list(lanelets)
        for lnlet in lanelets:
            if lnlet in final:
                (parallel, _, _) = self._get_parallel_lanelets(lnlet)
                final = [l for l in final if l not in parallel or l == lnlet]
        return set(final)

    def _reach_set_difference(self, a, b_dict):
        """
        Calculate the difference between two reachable sets.

        Subtracts b from a.
        """
        rs_list_pygeos = []

        a_poly_pygeos = [a[step] for step in a]
        len_max = max(len(a[step]) for step in a)
        a_poly_pygeos = polygon_padding(len_max, a_poly_pygeos)
        a_poly_pygeos = pygeos.polygons(a_poly_pygeos)

        # a_poly_pygeos_test = [pygeos.polygons(a[step]) for step in a]

        diff_pygeos = pygeos.difference(a_poly_pygeos, [b_dict[step] for step in a])
        key_list = list(a.keys())
        for i in range(len(key_list)):
            rs_list_pygeos += self._geom_to_reach_set(diff_pygeos[i], key_list[i])
        return rs_list_pygeos

    def _add_safe_distance(self, rs, rs_obj, safe_distance):
        """
        Extend a reachable set in longitudinal direction.

        Extend a reachable set in longitudinal direction.
        Applys the two-second heuristic for safe distances.
        """
        # move safe_distance calculation out to caller function
        # # 2-second safe distance heuristic
        # # urban scenarios (< 30 km/h)
        # if velocity <= 8:
        #     safe_distance_factor = 0.75
        # # built-up area / residential area (< 54 km/h)
        # elif velocity <= 15:
        #     safe_distance_factor = 1.0
        # # freeway (>= 54 km/h)
        # else:
        #     safe_distance_factor = 2.0
        # min_safe_distance = safe_distance_factor * velocity
        # safe_distance = min_safe_distance * safe_distance_frac

        # # Lane bounds used for reach set
        # patch = rs_obj.intersection_patch_pygeos
        # extended = {}
        # for step in rs:
        #     poly = Polygon(rs[step])
        #     buffer = Polygon(poly.buffer(safe_distance).exterior)
        #     if patch is not None:
        #         # trim extended with patch
        #         intersection = patch.intersection(buffer)
        #         if intersection.geom_type == "Polygon" and not intersection.is_empty:
        #             # convert intersection poly to points
        #             outline = list(zip(*intersection.exterior.coords.xy))
        #             extended[step] = np.array(outline)
        #         else:
        #             # intersection is no polygon
        #             # extended[step] = rs[step]
        #             raise ValueError(
        #                 "Unhandled geometry type: " + repr(intersection.geom_type)
        #             )
        #     elif buffer.geom_type == "Polygon" and not buffer.is_empty:
        #         outline = list(zip(*buffer.exterior.coords.xy))
        #         extended[step] = np.array(outline)
        #     else:
        #         extended[step] = rs[step]

# replace shapely operations by pygeos
        # Lane bounds used for reach set
        patch = rs_obj.intersection_patch_pygeos
        extended = {}
        poly_pygeos = [rs[step] for step in rs]
        len_max = max(len(rs[step]) for step in rs)
        poly_pygeos = polygon_padding(len_max, poly_pygeos)
        poly_pygeos = pygeos.polygons(poly_pygeos)
        # poly_pygeos = [pygeos.polygons(rs[step]) for step in rs]
        buffer_pygeos = pygeos.polygons(pygeos.get_exterior_ring(pygeos.buffer(poly_pygeos, safe_distance, quadsegs=16)))
        key_list = list(rs.keys())
        if patch is not None:
            intersection_pygeos = pygeos.intersection(patch, buffer_pygeos)
            for i in range(len(key_list)):
                # trim extended with patch
                # if intersection.geom_type == "Polygon" and not intersection.is_empty:
                if pygeos.get_type_id(intersection_pygeos[i]) == 3 and not pygeos.is_empty(intersection_pygeos[i]):
                    # convert intersection poly to points
                    outline = pygeos.get_coordinates(pygeos.get_exterior_ring(intersection_pygeos[i]))
                    extended[key_list[i]] = outline
                else:
                    # raise ValueError(
                    #     "Unhandled geometry type: " + repr(intersection.geom_type)
                    # )
                    raise ValueError(
                        "Unhandled geometry type: " + str(pygeos.get_type_id(intersection_pygeos[i]))
                    )
        else:
            for i in range(len(key_list)):
                if pygeos.get_type_id(buffer_pygeos[i]) == 3 and not pygeos.is_empty(buffer_pygeos[i]):
                    outline = pygeos.get_coordinates(pygeos.get_exterior_ring(buffer_pygeos[i]))
                    extended[key_list[i]] = outline
                else:
                    extended[key_list[i]] = rs[key_list[i]]
        return extended

    def _geom_to_reach_set(self, geometry, step):
        """
        Convert shapely geometry to reachable set.

        Convert shapely geometry to reachable set with only evaluation at step.
        """
        rs_list = []
        # if polygons don't intersect, the result is a MultiPolygon
        # if geometry.geom_type == "Polygon":
        if pygeos.get_type_id(geometry) == 3:
            if pygeos.is_empty(geometry):
                return rs_list
            # convert difference to points
            rs = {}
            rs[step] = self._get_points_of_polygon(geometry)
            rs_list.append(rs)
        # elif geometry.geom_type == "MultiPolygon":
        elif pygeos.get_type_id(geometry) == 6:
            for i in range(pygeos.get_num_geometries(geometry)):
                rs_list += self._geom_to_reach_set(pygeos.get_geometry(geometry, i), step)
        else:
            # raise ValueError("Unhandled geometry type: " + repr(geometry.geom_type))
            raise ValueError("Unhandled geometry type: " + str(pygeos.get_type_id(geometry)))
        return rs_list

    def _get_points_of_polygon(self, polygon):
        """
        Convert shapely polygon to numpy array.

        Convert a shapely polygon to a numpy array
        with columns [x,y].
        """
        # replace polygon coordinates extraction for shapely by pygeos
        # interior_line = []
        # for interior in polygon.interiors:
        #     interior_line = list(zip(*interior.coords.xy))
        # outline = list(zip(*polygon.exterior.coords.xy))
        # return np.array(outline + interior_line)
        interior_line = []
        for i in range(pygeos.get_num_interior_rings(polygon)):
            interior_line = pygeos.get_coordinates(pygeos.get_interior_ring(polygon, i)).tolist()
        outline = pygeos.get_coordinates(pygeos.get_exterior_ring(polygon)).tolist()
        return np.array(outline + interior_line)

    def _ego_reach_set(self, ego_state):
        """
        Compute the safe distance polygons for safe distance rule.

        Compute the reachable sets of the ego vehicle and extend it
        to account for the safe distance, resulting in the safe distance polygons.

        """
        l_id = find_lanelet_by_position_and_orientation(
            self.scenario.lanelet_network, ego_state.position, ego_state.orientation
        )[0]
        # bounds, ignoring laterally adjacent lanes
        bounds = self._calc_bounds_rec(
            lanelet_id=l_id,
            depth=self.reach_set_params["parameters"]["depth"],
            lateral=False,
        )
        self.reach_set_objs_single[l_id] = []
        for (l, r) in bounds:
            # reach set object trimmed to lanelet bounds
            obj = reachable_set_simple.ReachSetSimple(bound_l=l, bound_r=r)
            self.reach_set_objs_single[l_id].append(obj)

        self.ego_reach_set[ego_state.time_step] = []

        # call simple_reachable_set() before for loop, avoid unnecessary repeating
        # calculate reachable set
        srs = simple_reachable_set(
            obj_pos=ego_state.position,
            obj_heading=ego_state.orientation,
            obj_vel=ego_state.velocity,
            obj_length=self.ego_length,
            obj_width=self.ego_width,
            dt=self.reach_set_params["parameters"]["dt"],
            t_max=self.reach_set_params["parameters"]["t_max"],
            a_max=0.01,
        )
        srs_t = pygeos.polygons([srs[t_key] for t_key in srs.keys()])

        # calculate safe_distance before for loop
        # 2-second safe distance heuristic
        # urban scenarios (< 30 km/h)
        if ego_state.velocity <= 8:
            safe_distance_factor = 0.75
        # built-up area / residential area (< 54 km/h)
        elif ego_state.velocity <= 15:
            safe_distance_factor = 1.0
        # freeway (>= 54 km/h)
        else:
            safe_distance_factor = 2.0
        min_safe_distance = safe_distance_factor * ego_state.velocity
        safe_distance = min_safe_distance * self.reach_set_params["rules"]["safe_distance"]["safe_distance_frac"]

        for reach_set_obj in self.reach_set_objs_single[l_id]:
            # reach set for ego assumes constant acceleration
            # essentially, this is the center of the vehicle,
            # given its current velocity
            # rs = reach_set_obj.calc_reach_set(
            #     obj_pos=ego_state.position,
            #     obj_heading=ego_state.orientation,
            #     obj_vel=ego_state.velocity,
            #     obj_length=self.ego_length,
            #     obj_width=self.ego_width,
            #     dt=self.reach_set_params["parameters"]["dt"],
            #     t_max=self.reach_set_params["parameters"]["t_max"],
            #     a_max=0.01,
            # )
            rs = reach_set_obj.calc_reach_set(srs, srs_t)
            extended_rs = self._add_safe_distance(
                rs,
                reach_set_obj,
                safe_distance
            )
            self.ego_reach_set[ego_state.time_step].append(extended_rs)


def load_reach_set_json():
    """
    Load reachable_set.json with reach set parameters and rules.

    Returns:
        Dict: reach set parameters and rules from reachable_set.json
    """
    reach_set_config_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "reachable_set.json",
    )
    with open(reach_set_config_path, "r") as f:
        jsondata = json.load(f)

    return jsondata
