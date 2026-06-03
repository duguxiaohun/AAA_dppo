import pygame
import pygame.freetype
import weakref
import logging
import random
import collections
import numpy as np
import math
import cv2
import re
import sys
import os
from collections import deque, defaultdict

sys.path.append('/home/codon/CARLA/CARLA_0.9.12/PythonAPI/carla/dist/carla-0.9.12-py3.7-linux-x86_64.egg')
sys.path.append('/home/codon/CARLA/CARLA_0.9.12/PythonAPI/carla/')

import carla
from carla import ColorConverter as cc

CARLA_PYAPI = "/home/codon/CARLA/CARLA_0.9.12/PythonAPI/carla"
sys.path.insert(0, CARLA_PYAPI)
sys.path.insert(0, os.path.join(CARLA_PYAPI, "agents"))

from agents.navigation.basic_agent import BasicAgent
from carla import VehicleLightState as vls
import gym
from gym import spaces

screen_width, screen_height = 640, 360
WIDTH, HEIGHT, PACK = 80, 45, 4

# ── Scenario key coordinates — only edit here to move the scenario ────────────
# Junction: (~-82, -138).  Ego approaches from south (+y direction, northbound
# on screen), performs an UNPROTECTED LEFT TURN at the crossroads, exits EAST.
# Confirmed by drive_left_turn_test.py: going east (+x) IS the left turn.
_SPAWN_X, _SPAWN_Y, _SPAWN_Z = -84.5, -150.0, 0.3
_END1_X,  _END1_Y             = -50.0, -136.0   # inner eastbound lane
_END2_X,  _END2_Y             = -50.0, -139.5   # outer eastbound lane (~3.5 m offset)
# Approach/intersection threshold: y < -155 → approach; y >= -155 → junction+exit
# Set 17 m before junction so BasicAgent has time to plan the left turn
# (old -140 caused waypoint.next() to overshoot junction → straight instead of turn)
_JUNC_Y_THRESH = -155.0
# Derived bounds
_BOUND_X_MIN = min(_SPAWN_X, _END1_X, _END2_X) - 15
_BOUND_X_MAX = max(_SPAWN_X, _END1_X, _END2_X) + 15
_BOUND_Y_MIN = min(_SPAWN_Y, _END1_Y, _END2_Y) - 10
_BOUND_Y_MAX = max(_SPAWN_Y, _END1_Y, _END2_Y) + 10
# ─────────────────────────────────────────────────────────────────────────────


class InterSection(gym.Env):
    """
    Town03 left-turn crossroads scenario.
    Ego spawns south of junction (~-84.5, -160), drives north (+y direction),
    turns LEFT (east, +x) at the unprotected crossroads (~-82, -138).
    Two pre-computed smooth routes:
        wp  : spawn → inner eastbound lane  (town03_left_route1.npy)
        wp2 : spawn → outer eastbound lane  (town03_left_route2.npy)
    Routes are stored in env/map/town03/.
    """

    def __init__(self, enabled_obs_number=8, vehicle_type='single', use_checker=False,
                 control_interval=1, advanced_info=False,
                 surrounding_record=False, frame=10, port=2000,
                 seed=0, render=True, rangee=5,
                 randomize_obs_behavior=True,
                 obs_aggressiveness_range=(0.7, 1.0),
                 obs_speed_diff_range=(-80, -30),
                 obs_highway_speed_diff_range=(-20, 10),
                 obs_distance_range=(1, 2),
                 obs_max_speed_range=None,
                 obs_ignore_lights_range=(0, 0),
                 obs_ignore_signs_range=(0, 0),
                 obs_ignore_vehicles_range=(5, 20),
                 obs_auto_lane_change=True):

        self.image_size = WIDTH * HEIGHT
        self.action_size = 1
        self.spectator = None
        self.stop_step = 0

        self.vehicle_type = vehicle_type
        self.control_interval = control_interval
        self.advanced_info = advanced_info
        self.use_checker = use_checker
        self.surrounding_record = surrounding_record
        self.frame = frame
        self.rangee = rangee
        self.randomize_obs_behavior = randomize_obs_behavior
        self.obs_aggressiveness_range = tuple(obs_aggressiveness_range)
        self.obs_speed_diff_range = tuple(obs_speed_diff_range)
        self.obs_highway_speed_diff_range = tuple(obs_highway_speed_diff_range)
        self.obs_distance_range = tuple(obs_distance_range)
        self.obs_max_speed_range = None if obs_max_speed_range is None else tuple(obs_max_speed_range)
        self.obs_ignore_lights_range = tuple(obs_ignore_lights_range)
        self.obs_ignore_signs_range = tuple(obs_ignore_signs_range)
        self.obs_ignore_vehicles_range = tuple(obs_ignore_vehicles_range)
        self.obs_auto_lane_change = obs_auto_lane_change
        self.obs_behavior_dict = {}

        self.ego_vehicle = None
        self.obs_list, self.bp_obs_list, self.spawn_point_obs_list = [], [], []
        self.maximum_enabled_obs = 8
        self.enabled_obs = min(enabled_obs_number, self.maximum_enabled_obs)
        self.add_speed = 0

        self.collision_sensor = None
        self.seman_camera = None
        self.viz_camera = None
        self.surface = None
        self.camera_output = np.zeros([360, 640, 3])
        self.camera_output1 = np.zeros([360, 640, 3])
        self.recording = False
        self.Attachment = carla.AttachmentType
        self.obs_dim = 1
        self.action_dim = 2
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(self.action_dim,), dtype=np.float32
        )

        self.port = port
        self.client = carla.Client('localhost', port)
        self.client.set_timeout(10.0)

        self.world = self.client.load_world('Town03_Opt')
        self.world.unload_map_layer(carla.MapLayer.Buildings)
        self.map = self.world.get_map()

        self._weather_presets = find_weather_presets()
        self._weather_index = 8

        settings = self.world.get_settings()
        settings.no_rendering_mode = not (os.environ.get('CARLA_VISUALIZE', 'False').lower() == 'true')
        self.world.apply_settings(settings)

        self.seed = seed
        self.desire_speed = 5
        self.vehicle_state_dict = {}

    # ------------------------------------------------------------------
    def _sample_uniform(self, value_range):
        low, high = value_range
        return random.uniform(float(low), float(high))

    def _sample_int(self, value_range):
        low, high = value_range
        return random.randint(int(low), int(high))

    def _speed_diff_from_max_speed(self, vehicle, max_speed_kmh):
        speed_limit = max(vehicle.get_speed_limit(), 1.0)
        return 100.0 * (speed_limit - max_speed_kmh) / speed_limit

    def _sample_obs_behavior(self, vehicle, highway=False):
        if not self.randomize_obs_behavior:
            speed_range = self.obs_highway_speed_diff_range if highway else self.obs_speed_diff_range
            return {
                'aggressiveness': 0.0,
                'speed_diff': self._sample_int(speed_range),
                'distance': self._sample_int(self.obs_distance_range),
                'ignore_lights': self._sample_int(self.obs_ignore_lights_range),
                'ignore_signs': self._sample_int(self.obs_ignore_signs_range),
                'ignore_vehicles': self._sample_int(self.obs_ignore_vehicles_range),
            }

        aggressiveness = self._sample_uniform(self.obs_aggressiveness_range)
        if self.obs_max_speed_range is not None:
            max_speed_kmh = self._sample_uniform(self.obs_max_speed_range)
            speed_diff = self._speed_diff_from_max_speed(vehicle, max_speed_kmh)
        else:
            speed_range = self.obs_highway_speed_diff_range if highway else self.obs_speed_diff_range
            low, high = speed_range
            # Higher aggressiveness means faster target speed in TrafficManager
            # terms, i.e. a lower/possibly negative percentage difference.
            speed_diff = float(high) + aggressiveness * (float(low) - float(high))

        max_dist, min_dist = max(self.obs_distance_range), min(self.obs_distance_range)
        distance = max_dist + aggressiveness * (min_dist - max_dist)
        ignore_lights = self._sample_uniform(self.obs_ignore_lights_range)
        ignore_signs = self._sample_uniform(self.obs_ignore_signs_range)
        ignore_vehicles = self._sample_uniform(self.obs_ignore_vehicles_range)
        return {
            'aggressiveness': aggressiveness,
            'speed_diff': speed_diff,
            'distance': distance,
            'ignore_lights': ignore_lights,
            'ignore_signs': ignore_signs,
            'ignore_vehicles': ignore_vehicles,
        }

    def _apply_obs_behavior(self, vehicle, highway=False):
        behavior = self._sample_obs_behavior(vehicle, highway=highway)
        self.traffic_manager.auto_lane_change(vehicle, self.obs_auto_lane_change)
        self.traffic_manager.vehicle_percentage_speed_difference(vehicle, behavior['speed_diff'])
        self.traffic_manager.distance_to_leading_vehicle(vehicle, behavior['distance'])
        self.traffic_manager.ignore_lights_percentage(vehicle, behavior['ignore_lights'])
        self.traffic_manager.ignore_signs_percentage(vehicle, behavior['ignore_signs'])
        self.traffic_manager.ignore_vehicles_percentage(vehicle, behavior['ignore_vehicles'])
        self.obs_behavior_dict[vehicle.id] = behavior
        return behavior

    # ------------------------------------------------------------------
    def reset(self):
        self.vehicle_state_dict = {}
        self.obs_behavior_dict = {}
        self.stop_once = False
        self.stop_twice = False
        self.random = random.random()

        self.add_speed = 0
        self.stop_step = 0

        settings = self.world.get_settings()
        self.original_settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 1 / self.frame
        self.world.apply_settings(settings)

        self.steer_history = []
        self.intervene_history = []
        self.lat_action_history = []
        self.target_speed_history = []
        self.ego_location_history = deque(maxlen=10)
        self._steer_cache = 0
        self.ppp = None
        self.ppp1 = None
        self.y_aver = None
        self.dist_travelled = 0
        self.prev_dist_to_goal = None

        self.intervention = False
        self.risk = None
        self.v_upp = 19.5 / 7
        self.v_low = 13.5 / 7
        self.ii = None

        SpawnActor = carla.command.SpawnActor
        SetAutopilot = carla.command.SetAutopilot
        SetVehicleLightState = carla.command.SetVehicleLightState
        FutureActor = carla.command.FutureActor

        self.traffic_manager = self.client.get_trafficmanager(self.port + 500)
        self.traffic_manager.set_global_distance_to_leading_vehicle(1.0)
        self.traffic_manager.set_random_device_seed(self.seed)
        self.seed = (self.seed + 1) % 500
        self.traffic_manager.set_synchronous_mode(True)

        synchronous_master = False

        list_actor = self.world.get_actors()
        for actor_ in list_actor:
            if isinstance(actor_, carla.TrafficLight):
                actor_.set_state(carla.TrafficLightState.Green)
                actor_.set_green_time(2000.0)

        # ---- Load pre-computed smooth routes ----
        _map_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'map', 'town03')
        try:
            self.wp  = np.load(os.path.join(_map_dir, 'town03_left_route1.npy'))
            self.wp2 = np.load(os.path.join(_map_dir, 'town03_left_route2.npy'))
        except FileNotFoundError:
            raise FileNotFoundError(
                "town03_left_route1.npy / town03_left_route2.npy not found in env/map/town03/."
            )

        # Derive initial yaw from first two waypoints of route 1
        if len(self.wp) >= 2:
            dx = self.wp[1][0] - self.wp[0][0]
            dy = self.wp[1][1] - self.wp[0][1]
            _init_yaw = math.degrees(math.atan2(dy, dx))
        else:
            _init_yaw = 90.0

        # ---- Ego vehicle ----
        bp_ego = self.world.get_blueprint_library().filter('vehicle.mercedes.coupe_2020')[0]
        bp_ego.set_attribute('color', '0, 0, 0')
        bp_ego.set_attribute('role_name', 'hero')

        # rangee: spawn y in [_SPAWN_Y - rangee, _SPAWN_Y] (further south = further from junction)
        spawn_y = _SPAWN_Y - self.rangee * random.random()
        self._spawn_marker_xy = (_SPAWN_X, spawn_y)
        spawn_point_ego = carla.Transform(
            carla.Location(x=_SPAWN_X, y=spawn_y, z=_SPAWN_Z),
            carla.Rotation(yaw=_init_yaw),
        )

        if self.ego_vehicle is not None:
            self.destroy()
        self.ego_vehicle = self.world.spawn_actor(bp_ego, spawn_point_ego)

        self.ego_current_speed_ratio = -100
        self.traffic_manager.vehicle_percentage_speed_difference(
            self.ego_vehicle, self.ego_current_speed_ratio)

        self.target_speed = 48
        self.ego_vehicle.set_autopilot(True, self.traffic_manager.get_port())
        self.world.tick()

        self.agent = BasicAgent(self.ego_vehicle, target_speed=self.target_speed,
                                opt_dict={
                                    'ignore_vehicles': True,
                                    'dt': 1.0 / self.frame,   # match sim timestep (default 1/20 causes PID overshoot)
                                })
        self.ego_vehicle.set_autopilot(False)

        self.speed_limit_flag = 0
        l = self.ego_vehicle.bounding_box.extent.x * 0.9
        w = self.ego_vehicle.bounding_box.extent.y
        self.fix_theta = np.arctan(w / l) * 180 / np.pi
        self.fix_length = np.sqrt(l ** 2 + w ** 2)
        self.displacement_waypoint = self.map.get_waypoint(spawn_point_ego.location)
        self.waypoint_ego = self.map.get_waypoint(spawn_point_ego.location)

        self.count = 0
        self.subcount1 = 0
        self.subcount2 = 0
        self.interval = 2
        self.command_interval = round(self.frame / self.interval)
        self.list_action = []

        # ---- Surrounding vehicles (4 ports, 2-3 each, max 8 total) ----
        self.obs_list = []
        self.obs_velo_list = []
        self.obs_agent_list = []

        # Junction centre ≈ (-82, -138).
        # north_port      : southbound from north on N-S road  (yaw=270, opposing ego approach)
        # east_port       : westbound from east on E-W road    (yaw=180, left-turn conflict car)
        # west_port       : eastbound from west on E-W road    (yaw=0,   ahead of ego after exit)
        #                   x range capped at -115: road curves away beyond x≈-120 in Town03
        # west_west_port  : westbound near junction (yaw=180, right-turn direction traffic)
        #                   placed close to junction so they stay in scenario area
        north_port = [
            (-77.0, -108.0, 270.0),
            (-77.0,  -98.0, 270.0),
            (-77.0,  -88.0, 270.0),
        ]
        east_port = [
            (-40.0, -140.0, 180.0),
            (-30.0, -140.0, 180.0),
            (-20.0, -140.0, 180.0),
        ]
        west_port = [
            (-95.0,  -137.0, 0.0),
            (-105.0, -137.0, 0.0),
            (-115.0, -137.0, 0.0),
        ]
        west_west_port = [
            (-87.0,  -140.0, 180.0),
            (-92.0,  -140.0, 180.0),
            (-98.0,  -140.0, 180.0),
        ]

        spawn_points_by_port = []
        for port in [north_port, east_port, west_port, west_west_port]:
            n = random.randint(2, 3)
            chosen = random.sample(port, n)
            port_spawn_points = []
            for sx, sy, syaw in chosen:
                wp = self.map.get_waypoint(
                    carla.Location(x=sx, y=sy, z=0),
                    project_to_road=True,
                    lane_type=carla.LaneType.Driving)
                if wp is None:
                    continue
                loc = wp.transform.location
                port_spawn_points.append(carla.Transform(
                    carla.Location(x=loc.x, y=loc.y, z=loc.z + 0.3),
                    carla.Rotation(yaw=syaw),
                ))
            spawn_points_by_port.append(port_spawn_points)

        while sum(len(port) for port in spawn_points_by_port) > 8:
            removable_ports = [port for port in spawn_points_by_port if len(port) > 2]
            if not removable_ports:
                break
            selected_port = random.choice(removable_ports)
            selected_port.pop(random.randrange(len(selected_port)))

        spawn_points = []
        for port_spawn_points in spawn_points_by_port:
            spawn_points.extend(port_spawn_points)
        random.shuffle(spawn_points)
        spawn_points = spawn_points[:8]

        blueprints = self.world.get_blueprint_library().filter('vehicle.*')
        blueprints = [x for x in blueprints if (
            int(x.get_attribute('number_of_wheels')) == 4
            and x.id != 'vehicle.volkswagen.t2'
            and x.id != 'vehicle.bmw.isetta'
            and 'carlamotors' not in x.id
            and x.id == 'vehicle.tesla.model3'
        )]
        blueprints = sorted(blueprints, key=lambda bp: bp.id)

        batch = []
        for transform in spawn_points:
            bp_sv = random.choice(blueprints)
            if bp_sv.has_attribute('color'):
                color = random.choice(bp_sv.get_attribute('color').recommended_values)
                bp_sv.set_attribute('color', color)
            if bp_sv.has_attribute('driver_id'):
                driver_id = random.choice(bp_sv.get_attribute('driver_id').recommended_values)
                bp_sv.set_attribute('driver_id', driver_id)
            bp_sv.set_attribute('role_name', 'autopilot')
            light_state = vls.RightBlinker | vls.LeftBlinker | vls.Brake
            batch.append(SpawnActor(bp_sv, transform)
                         .then(SetAutopilot(FutureActor, True, self.traffic_manager.get_port()))
                         .then(SetVehicleLightState(FutureActor, light_state)))

        for response in self.client.apply_batch_sync(batch, synchronous_master):
            if response.error:
                logging.error(response.error)
            else:
                self.obs_list.append(response.actor_id)

        self.obs_actors = self.world.get_actors(self.obs_list)

        iii = 0
        for v in self.obs_actors:
            self._apply_obs_behavior(v)
            iii += 1

        self.speed_limit_obs_flags = np.zeros(iii)

        # ---- Collision sensor ----
        self.collision_history = []
        bp_collision = self.world.get_blueprint_library().find('sensor.other.collision')
        if self.collision_sensor is not None:
            self.collision_sensor.destroy()
        self.collision_sensor = self.world.spawn_actor(
            bp_collision, carla.Transform(), attach_to=self.ego_vehicle)
        weak_self = weakref.ref(self)
        self.collision_sensor.listen(lambda event: InterSection._on_collision(weak_self, event))

        self.count = 0
        self.count_yaw = 0
        self.count_speed = 0
        self.reset_traj_dataset()

        self.wp_idx  = 0
        self.wp2_idx = 0

        self.spect_cam_follow()

        self.agent.set_destination(self._make_target(self.wp[-1]))

        state = self.get_observation_scene()
        self.agent.ignore_vehicles(active=True)
        return state

    # ------------------------------------------------------------------
    def _on_collision(weak_self, event):
        self = weak_self()
        if not self:
            return
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x ** 2 + impulse.y ** 2 + impulse.z ** 2)
        self.collision_history.append((event.frame, intensity))
        if len(self.collision_history) > 4000:
            self.collision_history.pop(0)

    def get_collision_history(self):
        collision_history = collections.defaultdict(int)
        flag = 0
        for frame, intensity in self.collision_history:
            collision_history[frame] += intensity
            if intensity != 0:
                flag = 1
        return collision_history, flag

    # ------------------------------------------------------------------
    def get_position(self, waypoint):
        loc = waypoint.transform.location
        return [(loc.x, loc.y)]

    def depth_first_search(self, curr_waypoint, depth=0, max_depth=49):
        if depth > max_depth:
            return [self.get_position(curr_waypoint)]
        trasversed_lanes = []
        child_lanes = curr_waypoint.next(0.5)
        if len(child_lanes) > 0:
            for child in child_lanes:
                trajs = self.depth_first_search(child, depth + 1, max_depth)
                trasversed_lanes.extend(trajs)
        if len(trasversed_lanes) == 0:
            return [self.get_position(curr_waypoint)]
        res = []
        for lane in trasversed_lanes:
            res.append(self.get_position(curr_waypoint) + lane)
        return res

    def filter_and_pad(self, all_results, vehicle_location, k=3, length=50):
        lane_position = {}
        for i, result in enumerate(all_results):
            lane_position[i] = np.min(
                np.linalg.norm(np.array(result) - np.array(vehicle_location)[np.newaxis, :]))
        sort_lanes = sorted(lane_position.items(), key=lambda x: x[1])[:k]
        new_result = np.zeros((k, length, 2))
        for i, lane in enumerate(sort_lanes):
            select_lane = np.array(all_results[lane[0]])[:length]
            new_result[i] = np.pad(select_lane,
                                   pad_width=[[0, length - select_lane.shape[0]], [0, 0]])
        return new_result

    def fitler_goal_waypoints(self, results, goal, preview_dis):
        goal = np.array(goal)[np.newaxis, :]
        min_dist = [np.min(np.linalg.norm(np.array(r) - goal, axis=-1)) for r in results]
        arg = np.argmin(np.array(min_dist))
        return results[arg][preview_dis]

    def filter_initial_waypoints(self, result, ego_location, preview_dis):
        ego_location = np.array(ego_location)[np.newaxis, :]
        m_dist = np.argmin(np.linalg.norm(np.array(result) - ego_location, axis=-1))
        self.ego_wp = self.filter_and_pad([result], ego_location)
        return result[m_dist + preview_dis]

    def filter_planned_ego_waypoints(self, vehicle, preview_dis):
        location = vehicle.get_location()
        ego_xy = np.array([location.x, location.y])

        seg1 = self.wp [self.wp_idx :]
        seg2 = self.wp2[self.wp2_idx:]

        if len(seg1) > 0:
            local1 = int(np.argmin(np.linalg.norm(seg1 - ego_xy, axis=-1)))
            self.wp_idx = self.wp_idx + local1

        if len(seg2) > 0:
            local2 = int(np.argmin(np.linalg.norm(seg2 - ego_xy, axis=-1)))
            self.wp2_idx = self.wp2_idx + local2

        wp_pt  = self.wp [min(self.wp_idx  + preview_dis, len(self.wp)  - 1)]
        wp2_pt = self.wp2[min(self.wp2_idx + preview_dis, len(self.wp2) - 1)]

        d1 = np.linalg.norm(self.wp [self.wp_idx]  - ego_xy)
        d2 = np.linalg.norm(self.wp2[self.wp2_idx] - ego_xy)
        r  = wp_pt if d1 <= d2 else wp2_pt

        return wp_pt, r, wp2_pt

    def filter_ego_waypoints(self, vehicle, preview_dis):
        location = vehicle.get_location()
        waypoint = self.map.get_waypoint(location)
        vehicle_location = [location.x, location.y]

        goal = [_END2_X, _END2_Y]
        results = self.depth_first_search(waypoint, max_depth=200)

        # Approach phase: ego still south of junction (y < _JUNC_Y_THRESH)
        if vehicle_location[1] < _JUNC_Y_THRESH:
            r = self.filter_initial_waypoints(self.wp, vehicle_location, preview_dis)
        else:
            r = self.fitler_goal_waypoints(results, goal, preview_dis)

        lr, rr = None, None
        if (waypoint.lane_change & carla.LaneChange.Left != 0) and (waypoint.get_left_lane() is not None):
            if vehicle_location[1] < _JUNC_Y_THRESH:
                lr = self.filter_initial_waypoints(self.wp, vehicle_location, preview_dis)
            else:
                left_results = self.depth_first_search(waypoint.get_left_lane(), max_depth=200)
                lr = self.fitler_goal_waypoints(left_results, goal, preview_dis)

        if (waypoint.lane_change & carla.LaneChange.Right != 0) and (waypoint.get_right_lane() is not None):
            if vehicle_location[1] < _JUNC_Y_THRESH:
                rr = self.filter_initial_waypoints(self.wp, vehicle_location, preview_dis)
            else:
                right_results = self.depth_first_search(waypoint.get_right_lane(), max_depth=200)
                rr = self.fitler_goal_waypoints(right_results, goal, preview_dis)

        return lr, r, rr

    def get_all_waypoints(self, vehicle, judge=False):
        location = vehicle.get_location()
        waypoint = self.map.get_waypoint(location)

        results = self.depth_first_search(waypoint)
        if judge:
            self.judge_off_route(location.x, location.y)
        if (waypoint.lane_change & carla.LaneChange.Left != 0) and (waypoint.get_left_lane() is not None):
            results.extend(self.depth_first_search(waypoint.get_left_lane()))
        if (waypoint.lane_change & carla.LaneChange.Right != 0) and (waypoint.get_right_lane() is not None):
            results.extend(self.depth_first_search(waypoint.get_right_lane()))

        goal = [_END2_X, _END2_Y]
        return self.filter_and_pad(results, goal)

    # ------------------------------------------------------------------
    def select_top_actors(self, actors, vehicle_location, k=5):
        lane_position = {}
        for i, act in enumerate(actors):
            act_position = act.get_location()
            pos = [act_position.x, act_position.y]
            lane_position[i] = [np.linalg.norm(pos - np.array(vehicle_location)), 0]
        sort_lanes = sorted(lane_position.items(), key=lambda x: x[1][0])[:k]
        return sort_lanes

    def reset_traj_dataset(self):
        self.traj_dataset = defaultdict()
        self.traj_dataset['ego'] = dict()
        for obs_id in range(len(self.obs_actors)):
            self.traj_dataset['v_' + str(obs_id)] = dict()

    def angle_norm(self, yaw):
        theta = yaw - 90
        return (theta * np.pi / 180 + np.pi) % (2 * np.pi) - np.pi

    def get_actor_state(self, actor, types):
        return [actor.get_location().x, actor.get_location().y,
                self.angle_norm(actor.get_transform().rotation.yaw),
                actor.get_velocity().x, actor.get_velocity().y]

    def get_actor_state_total(self, actor, types):
        location = actor.get_location()
        velocity = actor.get_velocity()
        acceleration = actor.get_acceleration()
        steering = actor.get_control().steer
        return [location.x, location.y,
                velocity.x, velocity.y,
                acceleration.x, acceleration.y,
                steering,
                self.angle_norm(actor.get_transform().rotation.yaw)]

    def record_one_step(self):
        self.traj_dataset['ego'][self.count] = self.get_actor_state(self.ego_vehicle, 0)
        for obs_id in range(len(self.obs_actors)):
            self.traj_dataset['v_' + str(obs_id)][self.count] = \
                self.get_actor_state(self.obs_actors[obs_id], 1)

    def record_one_step_total(self):
        self.vehicle_state_dict['ego'] = self.get_actor_state_total(self.ego_vehicle, 0)
        for obs_id in range(len(self.obs_actors)):
            self.vehicle_state_dict['v_' + str(obs_id)] = \
                self.get_actor_state_total(self.obs_actors[obs_id], 1)

    def query_single_trajs(self, name):
        self_trajs = np.zeros((10, 5))
        queryed_trajs = self.traj_dataset[name]
        for i in range(10):
            queryed_time = self.count - i
            if queryed_time in queryed_trajs:
                self_trajs[-i, :] = np.array(queryed_trajs[queryed_time])
        return self_trajs

    def spect_cam_follow(self):
        if self.ego_vehicle is None:
            return
        self.spectator = self.world.get_spectator()
        ego_tf = self.ego_vehicle.get_transform()
        ego_loc = ego_tf.location
        ego_yaw = ego_tf.rotation.yaw
        cam_loc = carla.Location(x=ego_loc.x, y=ego_loc.y, z=ego_loc.z + 40.0)
        cam_rot = carla.Rotation(pitch=-90.0, yaw=ego_yaw, roll=0.0)
        self.spectator.set_transform(carla.Transform(cam_loc, cam_rot))

    def get_observation_scene(self):
        y_ego = self.ego_vehicle.get_location().y
        x_ego = self.ego_vehicle.get_location().x
        self.record_one_step()
        self.ego_location_history.append([x_ego, y_ego])
        if len(self.ego_location_history) == 1:
            step_dist = 0
        else:
            step_dist = np.sqrt(
                (self.ego_location_history[-2][0] - x_ego) ** 2 +
                (self.ego_location_history[-2][1] - y_ego) ** 2)
        self.dist_travelled += step_dist

        ego_traj = self.query_single_trajs('ego')
        select_actor_ids = self.select_top_actors(self.obs_actors, [x_ego, y_ego])
        neighbor_waypoints = np.zeros((6, 3, 50, 2))
        ego_waypoint = self.filter_and_pad([self.wp, self.wp2], [x_ego, y_ego])
        self.get_all_waypoints(self.ego_vehicle, judge=True)
        neighbor_waypoints[0] = ego_waypoint
        neighbor_trajs = np.zeros((6, 10, 5))
        neighbor_trajs[0] = ego_traj
        for i, actor_id in enumerate(select_actor_ids):
            actor_type = actor_id[1][1]
            index = actor_id[0]
            if actor_type == 0:
                actor = self.obs_actors[index]
                neighbor_waypoints[i + 1] = self.get_all_waypoints(actor)
                neighbor_trajs[i + 1] = self.query_single_trajs('v_' + str(index))

        neighbor_waypoints = neighbor_waypoints.reshape(18, 50, 2)
        return (neighbor_trajs, ego_traj[-1], neighbor_waypoints[:, ::5])

    # ------------------------------------------------------------------
    def action_adapter(self, model_action):
        speed = model_action[0]
        speed = (speed - (-1)) * (10 - 0) / (1 - (-1))
        speed = np.clip(speed, 0, 10)
        model_action[1] = np.clip(model_action[1], -1, 1)

        if model_action[1] < -1 / 3:
            lane = -1
        elif model_action[1] > 1 / 3:
            lane = 1
        else:
            lane = 0

        return (speed * 3.6, lane)

    def step(self, action):
        vx_ego = self.ego_vehicle.get_velocity().x
        vy_ego = self.ego_vehicle.get_velocity().y
        velocity_ego = (vx_ego ** 2 + vy_ego ** 2 +
                        self.ego_vehicle.get_velocity().z ** 2) ** 0.5
        self.count_speed += velocity_ego

        y_ego = self.ego_vehicle.get_location().y
        x_ego = self.ego_vehicle.get_location().x
        acceleration_ego = (self.ego_vehicle.get_acceleration().x ** 2 +
                            self.ego_vehicle.get_acceleration().y ** 2 +
                            self.ego_vehicle.get_acceleration().z ** 2) ** 0.5

        self.y_ego = y_ego
        self.x_ego = x_ego
        self.acceleration_ego = acceleration_ego
        self.vx_ego = vx_ego
        self.vy_ego = vy_ego
        self.velocity_ego = velocity_ego
        self.add_speed += velocity_ego

        self.world.tick()

        target_speed, lat_action = self.action_adapter(action[0])
        self.target_speed = target_speed
        self.agent.set_target_speed(self.target_speed)

        preview_dis = round(np.clip(velocity_ego * 2, 1, 15))
        self.filter_planned_ego_waypoints(self.ego_vehicle, preview_dis)

        nav_dis = max(preview_dis, 10)

        try:
            waypoint = self.map.get_waypoint(self.ego_vehicle.get_location())

            if y_ego < _JUNC_Y_THRESH:
                # ── Approach section (straight road, y < threshold) ──────────
                nexts = waypoint.next(nav_dis)
                if nexts:
                    self.agent.set_destination(nexts[0].transform.location)
            else:
                # ── Junction + exit section: cap speed so the sharp left turn
                #    doesn't overshoot the lane ──────────────────────────────
                junction_speed = min(self.target_speed, 25.0)
                self.agent.set_target_speed(junction_speed)
                if lat_action == 1:
                    dest = self.wp2[-1]   # outer lane
                else:
                    dest = self.wp[-1]    # inner lane (default)
                self.agent.set_destination(self._make_target(dest))
        except Exception:
            pass

        v_index = 0
        for v in self.obs_actors:
            if v.get_speed_limit() > 80 and self.speed_limit_obs_flags[v_index] == 0:
                self._apply_obs_behavior(v, highway=True)
                self.speed_limit_obs_flags[int(v_index)] = 1
            v_index += 1

        self.control = self.agent.run_step()
        y_ego = self.ego_vehicle.get_location().y
        x_ego = self.ego_vehicle.get_location().x
        vel = self.ego_vehicle.get_velocity()
        velocity_ego = (vel.x ** 2 + vel.y ** 2 + vel.z ** 2) ** 0.5

        self.ego_vehicle.apply_control(self.control)

        next_state = self.get_observation_scene()
        self.spect_cam_follow()

        self.collision = self.get_collision_history()[1]

        pos = np.array([x_ego, y_ego])
        self.finish = (
            np.linalg.norm(pos - np.array([_END1_X, _END1_Y])) < 15.0 or
            np.linalg.norm(pos - np.array([_END2_X, _END2_Y])) < 15.0
        )
        self.max_time = self.count > 250

        success = 2 if self.finish else 0
        coll = -2 if self.collision else 0
        coll = coll - 1 if self.off_route else coll
        coll = coll - 1 if self.max_time else coll

        pos = np.array([x_ego, y_ego])
        d1 = math.sqrt((x_ego - _END1_X) ** 2 + (y_ego - _END1_Y) ** 2)
        d2 = math.sqrt((x_ego - _END2_X) ** 2 + (y_ego - _END2_Y) ** 2)
        curr_dist = min(d1, d2)
        if self.prev_dist_to_goal is None:
            progress_reward = 0.0
        else:
            progress_reward = (self.prev_dist_to_goal - curr_dist) * 0.1
        self.prev_dist_to_goal = curr_dist

        self.record_one_step_total()

        if self.finish or self.collision or self.off_route or self.max_time:
            done = True
            if self.finish:
                print("finish is True")
            elif self.collision:
                print("collision is True")
            elif self.off_route:
                print("off_route is True")
            elif self.max_time:
                print("max_time is True")
        else:
            done = False

        speed_reward = velocity_ego
        reward = success + coll + 0.2 * speed_reward + progress_reward

        info = (
            self.finish, self.collision, self.off_route, self.max_time,
            self.count, self.count_speed / (self.count + 1), self.vehicle_state_dict
        )
        self.count += 1

        if done:
            self.destroy()

        return next_state, reward, done, info

    def return_vehicle_state(self):
        return self.vehicle_state_dict

    # ------------------------------------------------------------------
    def judge_off_route(self, x, y):
        ego_xy = np.array([[x, y]])
        d1 = np.min(np.linalg.norm(self.wp  - ego_xy, axis=-1))
        d2 = np.min(np.linalg.norm(self.wp2 - ego_xy, axis=-1))
        self.off_route = (min(d1, d2) > 6.0)

        if x < _BOUND_X_MIN or x > _BOUND_X_MAX:
            self.off_route = True
        if y < _BOUND_Y_MIN or y > _BOUND_Y_MAX:
            self.off_route = True

    # ------------------------------------------------------------------
    def destroy(self):
        self.collision_sensor.stop()
        actors = [self.ego_vehicle, self.collision_sensor]
        self.client.apply_batch_sync([carla.command.DestroyActor(x) for x in actors])
        self.client.apply_batch([carla.command.DestroyActor(x) for x in self.obs_list])
        self.collision_sensor = None
        self.ego_vehicle = None

    def IDM(self, index, y_target=None, v_target=None):
        delta = 4
        a, b, T, s0, v0 = 2.22, 1.67, 0.5, 5, 40
        if index is not None:
            s_delta = y_target - self.y_ego - 4
            v_delta = self.vy_ego - v_target
            s_prime = s0 + self.y_ego * T + self.vy_ego * v_delta / 2 / np.sqrt(a * b)
            acc = a * (1 - (self.vy_ego / v0) ** delta - (s_prime / s_delta) ** 2)
            target_speed = self.target_speed + acc * (1 / self.frame)
            target_speed = np.clip(target_speed, 36, 54)
        else:
            target_speed = v0
        return target_speed

    def _make_target(self, xy):
        x, y = float(xy[0]), float(xy[1])
        road_wp = self.map.get_waypoint(carla.Location(x, y, 0))
        return road_wp.transform.location

    def visualize_waypoints(self, life_time=120.0):
        """Draw wp (green) and wp2 (cyan) plus START/END markers in CARLA."""
        for point in self.wp:
            x, y = float(point[0]), float(point[1])
            z = self.map.get_waypoint(carla.Location(x, y, 0)).transform.location.z + 0.2
            self.world.debug.draw_point(
                carla.Location(x=x, y=y, z=z),
                size=0.08,
                color=carla.Color(0, 255, 0),
                life_time=life_time,
            )
        for point in self.wp2:
            x, y = float(point[0]), float(point[1])
            z = self.map.get_waypoint(carla.Location(x, y, 0)).transform.location.z + 0.2
            self.world.debug.draw_point(
                carla.Location(x=x, y=y, z=z),
                size=0.08,
                color=carla.Color(0, 255, 255),
                life_time=life_time,
            )

        def _z(x, y):
            return self.map.get_waypoint(carla.Location(x, y, 0)).transform.location.z

        sx, sy = getattr(self, '_spawn_marker_xy', (_SPAWN_X, _SPAWN_Y))
        self.world.debug.draw_point(carla.Location(x=sx, y=sy, z=_z(sx, sy)+1.0),
                                    size=0.4, color=carla.Color(255,255,0), life_time=life_time)
        self.world.debug.draw_string(carla.Location(x=sx, y=sy, z=_z(sx,sy)+3.5), 'START',
                                     color=carla.Color(255,255,0), life_time=life_time)

        e1x, e1y = _END1_X, _END1_Y
        self.world.debug.draw_point(carla.Location(x=e1x, y=e1y, z=_z(e1x,e1y)+1.0),
                                    size=0.4, color=carla.Color(0,255,0), life_time=life_time)
        self.world.debug.draw_string(carla.Location(x=e1x, y=e1y, z=_z(e1x,e1y)+3.5), 'END1',
                                     color=carla.Color(0,255,0), life_time=life_time)

        e2x, e2y = _END2_X, _END2_Y
        self.world.debug.draw_point(carla.Location(x=e2x, y=e2y, z=_z(e2x,e2y)+1.0),
                                    size=0.4, color=carla.Color(0,255,255), life_time=life_time)
        self.world.debug.draw_string(carla.Location(x=e2x, y=e2y, z=_z(e2x,e2y)+3.5), 'END2',
                                     color=carla.Color(0,255,255), life_time=life_time)

    def _next_weather(self, reverse=False):
        self._weather_index += -1 if reverse else 1
        self._weather_index %= len(self._weather_presets)
        preset = self._weather_presets[self._weather_index]
        self.world.set_weather(preset[0])


def find_weather_presets():
    rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
    name = lambda x: ' '.join(m.group(0) for m in rgx.finditer(x))
    presets = [x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)]
    return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]
