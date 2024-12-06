from track_handler_py import Track, Raceline
import numpy as np
from dataclasses import dataclass
import param_management_py as pmg


@dataclass(init=False)
class CalculationCostsParams():

    curvature_cost_weight: float
    curvature_cost_threshold: float
    raceline_cost_weight: float
    velocity_cost_weight: float
    friction_cost_weight: float
    lateral_jerk_cost_weight: float
    raceline_cost_weight_overtaking: float
    velocity_cost_weight_overtaking: float
    lateral_jerk_cost_weight_overtaking: float
    prediction_cost_weight: float 
    additional_absolute_sample_cost: float
    collision_cost_weight: float
    horizon: float 
    prediction_s_factor_min_size: float 
    prediction_s_factor_max_size: float
    prediction_n_factor: float
    prediction_s_factor_defender: float
    prediction_n_factor_defender: float
    prediction_s_factor_static: float
    prediction_n_factor_static: float
    prediction_uncertainty_weight: float
    increasing_rl_cost: bool
    scale_horizon: float
    velocity_excess_cost_multiplier: float
    s_positions_loc_uncertain: bool
    const_intervals_loc_uncertain: bool
    velocity_loc_uncertain: bool
    fading_distance_loc_uncertain: bool
    max_deceleration_on_target_change: float
    V_diff_max_costs: float
    collision_check_horizon: int
    vehicle_length: float
    vehicle_width: float
    tube_width: float
    safety_distance_vehicles: float
    
class CalculationCosts():
    def __init__(self, param_manager, debugging):
        self.param_manager = param_manager
        self.params = CalculationCostsParams()
        self.param_state_hash = None
        self.debugging = debugging
        self.declare_and_update_parameters()

    def declare_and_update_parameters(self):
        if self.param_manager.get_state_hash() != self.param_state_hash:
            self.params.curvature_cost_weight = self.param_manager.declare_and_get_value("costs.curvature_cost_weight", 500000.0, pmg.ParameterType.DOUBLE, "curvature cost weigh").as_double()
            self.params.curvature_cost_threshold = self.param_manager.declare_and_get_value("costs.curvature_cost_threshold", 30.0, pmg.ParameterType.DOUBLE, "curvature cost threshold").as_double()
            self.params.raceline_cost_weight = self.param_manager.declare_and_get_value("costs.raceline_cost_weight", 3.5, pmg.ParameterType.DOUBLE, "raceline cost weight").as_double()
            self.params.velocity_cost_weight = self.param_manager.declare_and_get_value("costs.velocity_cost_weight", 3.0, pmg.ParameterType.DOUBLE, "velocity cost weight").as_double()
            self.params.friction_cost_weight = self.param_manager.declare_and_get_value("costs.friction_cost_weight", 5000.0, pmg.ParameterType.DOUBLE, "friction cost weight").as_double()
            self.params.lateral_jerk_cost_weight = self.param_manager.declare_and_get_value("costs.lateral_jerk_cost_weight", 0.0, pmg.ParameterType.DOUBLE, "lateral jerk cost weight").as_double()
            self.params.raceline_cost_weight_overtaking = self.param_manager.declare_and_get_value("overtaking_weights.raceline_cost_weight_overtaking", 2.0, pmg.ParameterType.DOUBLE, "racline_cost_weight").as_double()
            self.params.velocity_cost_weight_overtaking = self.param_manager.declare_and_get_value("overtaking_weights.velocity_cost_weight_overtaking", 8.0, pmg.ParameterType.DOUBLE, "velocity cost weight overtaking").as_double()
            self.params.lateral_jerk_cost_weight_overtaking = self.param_manager.declare_and_get_value("overtaking_weights.lateral_jerk_cost_weight_overtaking", 0.0, pmg.ParameterType.DOUBLE, "lateral jerk cost weight overtaking").as_double()
            self.params.prediction_cost_weight = self.param_manager.declare_and_get_value("costs.prediction_cost_weight", 100000.0, pmg.ParameterType.DOUBLE, "prediction cost weight").as_double()
            self.params.additional_absolute_sample_cost = self.param_manager.declare_and_get_value("costs.additional_absolute_sample_cost", 50.0, pmg.ParameterType.DOUBLE, "additional absolute sample cost").as_double()
            self.params.collision_cost_weight = self.param_manager.declare_and_get_value("costs.collision_cost_weight", 100000000.0, pmg.ParameterType.DOUBLE, "collision cost weight").as_double()
            self.params.horizon = self.param_manager.declare_and_get_value("behavior.horizon", 4.0, pmg.ParameterType.DOUBLE, "Planning Horizon").as_double()
            self.params.prediction_s_factor_min_size = self.param_manager.declare_and_get_value("costs.prediction_s_factor_min_size", 0.03, pmg.ParameterType.DOUBLE, "prediction s factor min size").as_double()
            self.params.prediction_s_factor_max_size = self.param_manager.declare_and_get_value("costs.prediction_s_factor_max_size", 0.012, pmg.ParameterType.DOUBLE, "prediction s factor max size").as_double()
            self.params.prediction_n_factor = self.param_manager.declare_and_get_value("costs.prediction_n_factor", 0.2, pmg.ParameterType.DOUBLE, "prediction n factor").as_double()
            self.params.prediction_s_factor_defender = self.param_manager.declare_and_get_value("costs.prediction_s_factor_defender", 0.03, pmg.ParameterType.DOUBLE, "prediction s factor defender").as_double()
            self.params.prediction_n_factor_defender = self.param_manager.declare_and_get_value("costs.prediction_n_factor_defender", 0.2, pmg.ParameterType.DOUBLE, "prediction n factor defender").as_double()
            self.params.prediction_s_factor_static = self.param_manager.declare_and_get_value("costs.prediction_s_factor_static", 0.05, pmg.ParameterType.DOUBLE, "prediction s factor static").as_double()
            self.params.prediction_n_factor_static = self.param_manager.declare_and_get_value("costs.prediction_n_factor_static", 0.35, pmg.ParameterType.DOUBLE, "prediction n factor static").as_double()
            self.params.prediction_uncertainty_weight = self.param_manager.declare_and_get_value("costs.prediction_uncertainty_weight", 4.0, pmg.ParameterType.DOUBLE, "prediction uncertainty weight").as_double()
            self.params.increasing_rl_cost = self.param_manager.declare_and_get_value("costs.increasing_rl_cost", True, pmg.ParameterType.BOOL, "increasing rl cost").as_bool()
            self.params.scale_horizon = self.param_manager.declare_and_get_value("costs.scale_horizon", 1.0, pmg.ParameterType.DOUBLE, "scale horizon").as_double()
            self.params.velocity_excess_cost_multiplier = self.param_manager.declare_and_get_value("costs.velocity_excess_cost_multiplier", 2.0, pmg.ParameterType.DOUBLE, "s positions loc uncertainty").as_double()
            self.params.s_positions_loc_uncertain = self.param_manager.declare_and_get_value("distances.s_positions_loc_uncertain", False, pmg.ParameterType.BOOL, "s positions loc uncertainty").as_bool()
            self.params.const_intervals_loc_uncertain = self.param_manager.declare_and_get_value("distances.const_intervals_loc_uncertain", False, pmg.ParameterType.BOOL, "const intervals loc uncertainty").as_bool()
            self.params.velocity_loc_uncertain = self.param_manager.declare_and_get_value("distances.velocity_loc_uncertain", False, pmg.ParameterType.BOOL, "velocity loc uncertainty").as_bool()
            self.params.fading_distance_loc_uncertain = self.param_manager.declare_and_get_value("distances.fading_distance_loc_uncertain", False, pmg.ParameterType.BOOL, "fading distance loc uncertainty").as_bool()
            self.params.max_deceleration_on_target_change = self.param_manager.declare_and_get_value("behavior.max_deceleration_on_target_change", 5.0, pmg.ParameterType.DOUBLE, "max decerlation on target change").as_double()
            self.params.V_diff_max_costs = self.param_manager.declare_and_get_value("costs.V_diff_max_costs", 15.0, pmg.ParameterType.DOUBLE, "V diff max costs").as_double()
            self.params.collision_check_horizon = self.param_manager.declare_and_get_value("behavior.collision_check_horizon", 20, pmg.ParameterType.INTEGER, "Collision check horizon").as_int()
            self.params.vehicle_length = self.param_manager.declare_and_get_value("behavior.vehicle_length", 5.0, pmg.ParameterType.DOUBLE, "vehicle length").as_double()
            self.params.vehicle_width = self.param_manager.declare_and_get_value("behavior.vehicle_width", 2.0, pmg.ParameterType.DOUBLE, "vehicle width").as_double()
            self.params.tube_width = self.param_manager.declare_and_get_value("behavior.tube_width", 1.0, pmg.ParameterType.DOUBLE, "tube width").as_double()
            self.params.safety_distance_vehicles = self.param_manager.declare_and_get_value("behavior.safety_distance_vehicles", 0.0, pmg.ParameterType.DOUBLE, "safety distance vehicles").as_double()
            self.param_state_hash = self.param_manager.get_state_hash()


    def sort_trajectories_by_cost(self,
                                   valid_array,
                                   cost_array):
        # get sorted indices of costs, from lowest to highest
        sorted_idx = np.argsort(cost_array[valid_array])

        return sorted_idx

    def calc_costs(
            self,
            valid_array: np.ndarray,
            rel_long_sampling_array: np.ndarray,
            track_handler: Track,
            s_array: np.ndarray,
            n_array: np.ndarray,
            t_array: np.ndarray,
            V_array: np.ndarray,
            Omega_z_array: np.ndarray,
            ay_array: np.ndarray,
            raceline: dict,
            prediction: dict,
            V_target: float,
            planning_requests: dict,
            tire_util_array: np.ndarray,
            pitlane_mode: bool,
            vehicle_ahead: bool,
            emergency_brake: bool


    ) -> int:
        
        self.declare_and_update_parameters()

        curvature_cost_array = np.zeros_like(valid_array, dtype=float)
        lat_jerk_cost_array = np.zeros_like(valid_array, dtype=float)
        velocity_cost_array = np.zeros_like(valid_array, dtype=float)
        collision_cost_array = np.zeros_like(valid_array, dtype=float)
        raceline_cost_array = np.zeros_like(valid_array, dtype=float)
        friction_cost_array = np.zeros_like(valid_array, dtype=float)
        prediction_cost_array = np.zeros_like(valid_array, dtype=float)

        # store raw cost terms for debugging
        if self.debugging:
            curvature_cost_array_raw = np.zeros((len(valid_array), len(t_array[0])-1))
            lat_jerk_cost_array_raw = np.zeros((len(valid_array), len(t_array[0])-1))
            velocity_cost_array_raw = np.zeros((len(valid_array), len(t_array[0])-1))
            collision_cost_array_raw = np.zeros((len(valid_array), len(t_array[0])-1))
            raceline_cost_array_raw = np.zeros((len(valid_array), len(t_array[0])-1))
            friction_cost_array_raw = np.zeros((len(valid_array), len(t_array[0])-1))
            prediction_cost_array_raw = np.zeros((len(valid_array), len(t_array[0])-1))

        # general expressions
        exponent = self.params.scale_horizon - 1.0
        scale_horizon_factor = np.tile(
            np.linspace(0.1**exponent, 0.0 if exponent > -1.0 else 0.1**exponent, s_array[valid_array, :-1].shape[1])
            * np.logspace(exponent, 0.0 if exponent > -1.0 else exponent, s_array[valid_array, :-1].shape[1]),
            (s_array[valid_array, :-1].shape[0], 1),
        )

        increasing_time_factor = np.minimum(t_array[valid_array] / self.params.horizon, 1.0)
        diff_time_array = np.diff(t_array[valid_array], axis=1)
        V_fading_factor = np.ones_like(s_array[valid_array])

        V_raceline = np.interp(s_array[valid_array], raceline["s_post"], raceline["V_post"], period=track_handler.s_coord()[-1])

        # check if raceline available and correct map
        if raceline and raceline["pitlane"] == pitlane_mode:
            raceline_deviation = (
                np.interp(s_array[valid_array], raceline["s_post"], raceline["n_post"], period=track_handler.s_coord()[-1])
                - n_array[valid_array]
            )
            centerline_deviation = n_array[valid_array]
            if (
                self.params.s_positions_loc_uncertain
                and self.params.const_intervals_loc_uncertain
                and self.params.velocity_loc_uncertain
                and self.params.fading_distance_loc_uncertain
            ):
                for s_loc_uncertain, i_loc_uncertain, v_loc_uncertain, fade_loc_uncertain in zip(
                    self.params.s_positions_loc_uncertain,
                    self.params.const_intervals_loc_uncertain,
                    self.params.velocity_loc_uncertain,
                    self.params.fading_distance_loc_uncertain,
                ):
                    fading_factor = np.interp(
                        s_array[valid_array],
                        [
                            s_loc_uncertain - i_loc_uncertain - fade_loc_uncertain,
                            s_loc_uncertain - i_loc_uncertain,
                            s_loc_uncertain,
                            s_loc_uncertain + i_loc_uncertain,
                            s_loc_uncertain + i_loc_uncertain + fade_loc_uncertain,
                        ],
                        [0.0, 1.0, 1.0, 1.0, 0.0],
                    )
                    raceline_deviation = (
                        -fading_factor * centerline_deviation + (1.0 - fading_factor) * raceline_deviation
                    )
                    V_fading_factor *= fading_factor * v_loc_uncertain + (1.0 - fading_factor)
        else:
            raceline_deviation = n_array[valid_array]

        # set costs according to role and if overtaking is allowed
        # 0 = no flag, 1 = defender, 2 = attacker

        # Init cost weight with default params
        velocity_cost_weight = self.params.velocity_cost_weight
        raceline_cost_weight = self.params.raceline_cost_weight
        lateral_jerk_cost_weight = self.params.lateral_jerk_cost_weight

        # reduce safety zone size when in defender role 
        if planning_requests["role"] == 1:
            prediction_s_factor = self.params.prediction_s_factor_defender
            prediction_n_factor = self.params.prediction_n_factor_defender

        elif planning_requests["role"] == 2 and planning_requests["overtaking_allowed"]:
            velocity_cost_weight = self.params.velocity_cost_weight_overtaking
            raceline_cost_weight = self.params.raceline_cost_weight_overtaking
            lateral_jerk_cost_weight = self.params.lateral_jerk_cost_weight_overtaking

            # adjust longitudinal safety ellipse size to current velocity in atacker mode
            prediction_s_factor = np.interp(V_array[0][0], [15.0, 50.0], [self.params.prediction_s_factor_min_size, self.params.prediction_s_factor_max_size])
            prediction_n_factor = self.params.prediction_n_factor

        # take max longitudinal ellipse size if no vehicle flag is specified
        else:
            prediction_s_factor = self.params.prediction_s_factor_max_size
            prediction_n_factor = self.params.prediction_n_factor

        # ------------------------------------------------------------------------------------------------------------------
        # CURVATURE COST
        # ------------------------------------------------------------------------------------------------------------------
        # set curvature cost to zero when below preset speed
        curvature_rl = np.interp(
            s_array[valid_array], track_handler.s_coord(), track_handler.omega_z(), period=track_handler.s_coord()[-1]
        )
        curvature_diff_array = np.where(
            V_array[valid_array, :-1] < self.params.curvature_cost_threshold,
            0.0,
            (np.abs(Omega_z_array[valid_array, :-1]) - np.abs(curvature_rl[:, :-1])),
        )

        curvature_cost = curvature_diff_array**2

        curvature_cost_array[valid_array] = self.params.curvature_cost_weight * np.add.reduce(
            curvature_cost * diff_time_array * np.sqrt(V_array[valid_array, :-1]), axis=1
        )

        # ------------------------------------------------------------------------------------------------------------------
        # LATERAL JERK COST
        # ------------------------------------------------------------------------------------------------------------------
        lat_jerk_array = np.abs(np.diff(ay_array[valid_array]))
        lat_jerk_cost_array[valid_array] = lateral_jerk_cost_weight * np.add.reduce(
            lat_jerk_array * diff_time_array, axis=1
        )

        # ------------------------------------------------------------------------------------------------------------------
        # VELOCITY COST
        # ------------------------------------------------------------------------------------------------------------------
        # set target speed to zero if all edges collide and behind an opponent vehicle
        if emergency_brake and vehicle_ahead:
            V_target = 0.0
            V_diff_array = np.abs(np.minimum(V_raceline[:, :-1], V_target) - V_array[valid_array, :-1])
            
        # reduce deceleration when hard braking is not necessary
        # V_target + 1.0 is used to avoid numerical errors
        elif (V_target + 1.0) < V_array[0][0] and not self.params.max_deceleration_on_target_change <= 0:  # Deceleration required
            v_start = V_array[0][0]  # current velocity
            t_end = t_array[0][-1]
            v_end = (
                v_start - max(self.params.max_deceleration_on_target_change, 2.0) * t_end
            )  # avoid user error setting to low acceleration
            V_target_array = np.maximum(np.interp(t_array[0], [0, t_end], [v_start, v_end]), V_target)
            V_diff_array = V_array[valid_array, :-1] - V_target_array[:-1]
        # cap maximum delta speed when accelerating
        else:
            # normalize cost term
            V_diff_array_equal = np.minimum(V_raceline[:, :-1], V_target) - V_array[valid_array, :-1]
            # Mutiply all negative values in V_diff x velocity_excess_cost_multiplier
            V_diff_array_unequal = np.where(V_diff_array_equal >= 0.0, 1.0, self.params.velocity_excess_cost_multiplier) * V_diff_array_equal
            V_diff_array_unsaturated = np.abs(V_diff_array_unequal)
            V_diff_max_cur = np.max(V_diff_array_unsaturated)
            V_diff_array = V_diff_array_unsaturated / V_diff_max_cur * np.minimum(self.params.V_diff_max_costs, V_diff_max_cur)

        velocity_cost = V_diff_array ** 2

        velocity_cost_array[valid_array] = velocity_cost_weight * np.add.reduce(
            velocity_cost * diff_time_array, axis=1
        )

        # ------------------------------------------------------------------------------------------------------------------
        # RACELINE COST
        # ------------------------------------------------------------------------------------------------------------------
        # either equally weighted or linearly increasing over horizon
        raceline_cost = (raceline_deviation[:, :-1]**2) * increasing_time_factor[:, :-1] if self.params.increasing_rl_cost else np.abs(raceline_deviation[:, :-1])
        
        raceline_cost_array[valid_array] = raceline_cost_weight * np.add.reduce(
            raceline_cost * diff_time_array, axis=1)

        # ------------------------------------------------------------------------------------------------------------------
        # FRICTION COST
        # ------------------------------------------------------------------------------------------------------------------
        # add costs where tire limits are slightly violated
        friction_violation_array = np.where(tire_util_array[valid_array][:, :-1] <= 1.0, 0.0, tire_util_array[valid_array][:, :-1] ** 2)

        friction_cost_array[valid_array] = self.params.friction_cost_weight * np.add.reduce(
                np.abs(friction_violation_array) * diff_time_array,
                axis=1
            )
            
        # ------------------------------------------------------------------------------------------------------------------
        # PREDICTION COST
        # ------------------------------------------------------------------------------------------------------------------

        # if no prediction reveived
        weighted_prediction_costs = np.zeros((len(valid_array), len(t_array[0])-1))[valid_array]

        for pred_idx, prediction_id in enumerate(prediction):
            prediction_cur = prediction[prediction_id]

            # check if prediction is considered at all
            if not prediction_cur["valid"]:
                continue

            # use smaller ellipse for static objects
            if prediction_cur["prediction_type"] == "static":
                prediction_s_factor = self.params.prediction_s_factor_static
                prediction_n_factor = self.params.prediction_n_factor_static

            # time factor reduced prediction costs that is further in the future
            time_uncertain = self.params.horizon
            time_pred_uncertainty = t_array[valid_array] + prediction_cur["time_offset"]
            time_factor = np.minimum(time_pred_uncertainty / time_uncertain, 1.0)

            # get distances to predicted vehicle and handle start-finish line
            s_prediction_cur = np.interp(
                t_array[valid_array],
                prediction_cur["time_w_offset"],
                prediction_cur["s"],
                period=track_handler.s_coord()[-1],
            )
            n_prediction_cur = np.interp(t_array[valid_array], prediction_cur["time_w_offset"], prediction_cur["n"])

            track_length = track_handler.s_coord()[-1]
            s_dist_temp = np.abs(s_array[valid_array] - s_prediction_cur)
            s_dist = np.where(s_dist_temp < track_length / 2.0, s_dist_temp, track_length - s_dist_temp)
            n_dist = np.abs(n_array[valid_array] - n_prediction_cur)

            # less weight to prediction points further into the future
            uncertainty_discount = np.exp(-self.params.prediction_uncertainty_weight * time_factor**2)

            raw_prediction_costs = np.exp(-prediction_s_factor * (s_dist) ** 2 - prediction_n_factor * (n_dist) ** 2)
            weighted_prediction_costs = raw_prediction_costs[:, :-1] * uncertainty_discount[:, :-1]
            prediction_cost_array[valid_array] += self.params.prediction_cost_weight * np.add.reduce(
                weighted_prediction_costs * diff_time_array, axis=1
            )

        # ------------------------------------------------------------------------------------------------------------------
        # COLLISION COST
        # ------------------------------------------------------------------------------------------------------------------

        # ensure that index is integer
        collision_check_horizon = int(self.params.collision_check_horizon)

        for pred_idx, prediction_id in enumerate(prediction):
            prediction_cur = prediction[prediction_id]
            if prediction_cur["valid"]:
                s_prediction_cur = np.interp(
                    t_array[valid_array][0], prediction_cur["time_w_offset"], prediction_cur["s"]
                )
                n_prediction_cur = np.interp(
                    t_array[valid_array][0], prediction_cur["time_w_offset"], prediction_cur["n"]
                )

                # select points to check for collision
                s_pred_check = s_prediction_cur[:collision_check_horizon]
                n_pred_check = n_prediction_cur[:collision_check_horizon]

                s_traj_check = s_array[valid_array, :collision_check_horizon]
                n_traj_check = n_array[valid_array, :collision_check_horizon]

                s_diff_tmp = np.abs(s_pred_check - s_traj_check)
                n_diff = np.abs(n_pred_check - n_traj_check)

                # handle start finish line
                track_length = track_handler.s_coord()[-1]
                s_diff = np.where(s_diff_tmp < track_length / 2.0, s_diff_tmp, track_length - s_diff_tmp)

                # handle start-finish line

                # check for collisions in s and n
                s_collision = s_diff < (self.params.vehicle_length + self.params.safety_distance_vehicles)
                n_collision = n_diff < (self.params.vehicle_width + self.params.tube_width + self.params.safety_distance_vehicles)

                # get earliest time stamp for a collision for each trajectory
                collision = (s_collision) & (n_collision)

                # store where collision occurs
                collision_mask = np.any(collision, axis=1)

                # handle empty arrays
                if collision.size == 0:
                    collision_cost_array = np.zeros_like(valid_array)
                else:
                    # some ugly lines to distinguish between collision in first step and no collision
                    earliest_idx = np.argmax(collision, axis=1)
                    earliest_idx = np.where(collision_mask, earliest_idx, -1)
                    time_to_collision = t_array[0][earliest_idx]

                    # velocity difference on projected impact
                    if prediction_cur["prediction_type"] != "static":
                        pred_vel = prediction_cur["vel"]
                        delta_vel_on_collision = np.where(earliest_idx == -1, 0.0, np.abs(pred_vel[earliest_idx] - V_array[valid_array, earliest_idx]))
                    else: # handle static predictions
                        delta_vel_on_collision = np.where(earliest_idx == -1, 0.0, np.abs(V_array[valid_array, earliest_idx]))

                    # prevent division by zero
                    time_to_collision = np.maximum(time_to_collision, 0.01)

                    collision_cost = delta_vel_on_collision / (time_to_collision ** 2)

                    collision_cost_array[valid_array] += self.params.collision_cost_weight * collision_cost

        # ------------------------------------------------------------------------------------------------------------------
        # PUNISHMENT FOR USING ABSOLUTE SAMPLES
        # ------------------------------------------------------------------------------------------------------------------
        velocity_cost_array[valid_array] = np.where(rel_long_sampling_array[valid_array], velocity_cost_array[valid_array], self.params.additional_absolute_sample_cost + velocity_cost_array[valid_array])
        # ------------------------------------------------------------------------------------------------------------------

        # store raw cost terms over time for debugging
        if self.debugging:
            curvature_cost_array_raw[valid_array] = curvature_cost
            lat_jerk_cost_array_raw[valid_array] = lat_jerk_array
            velocity_cost_array_raw[valid_array] = velocity_cost
            #collision_cost_array_raw[valid_array] = collision_cost
            raceline_cost_array_raw[valid_array] = raceline_cost
            friction_cost_array_raw[valid_array] = friction_violation_array
            prediction_cost_array_raw[valid_array] = weighted_prediction_costs

        # OVERALL COSTS
        cost_array = curvature_cost_array + velocity_cost_array + raceline_cost_array + prediction_cost_array + lat_jerk_cost_array + friction_cost_array + collision_cost_array

        if self.debugging:
            cost_extensive_array = [lat_jerk_cost_array_raw, velocity_cost_array_raw, raceline_cost_array_raw, prediction_cost_array_raw, lat_jerk_cost_array_raw, friction_cost_array_raw, None]
        else:
            cost_extensive_array = None

        return cost_array, [curvature_cost_array, velocity_cost_array, raceline_cost_array, prediction_cost_array, lat_jerk_cost_array, friction_cost_array, collision_cost_array], cost_extensive_array
