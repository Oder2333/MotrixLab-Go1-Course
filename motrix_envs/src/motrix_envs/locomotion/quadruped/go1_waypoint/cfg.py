# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Lesson 3: finite flat route, rule-generated commands, strict final stance."""

from dataclasses import replace
from pathlib import Path

import numpy as np
from motrix_robots.unitree import UnitreeGo1Robot

from motrix_env_core import registry
from motrix_env_core.base import SimCfg
from motrix_env_core.config import configclass
from motrix_env_core.config.scene import FlatTerrainCfg
from motrix_env_core.config.scene.base import SystemCameraCfg
from motrix_envs.config.scene import StandardSceneObjsCfg
from motrix_envs.locomotion.quadruped.cfg import NoiseConfig, QuadrupedSceneCfg
from motrix_envs.locomotion.quadruped.go1 import Go1WalkDirectEnvCfg

from .logic import DEFAULT_WAYPOINTS

ROOT = Path(__file__).resolve().parent


@configclass
class WaypointRouteCfg:
    waypoints: tuple[tuple[float, float], ...] = DEFAULT_WAYPOINTS
    spawn_xy: tuple[float, float] = (-3.0, -2.0)
    spawn_jitter: float = 0.15
    spawn_yaw_range: tuple[float, float] = (-0.35, 0.35)
    bounds: tuple[float, float, float, float] = (-5.0, 4.0, -4.0, 3.0)
    reach_radius: float = 0.35

    def contains(self, xy):
        xmin, xmax, ymin, ymax = self.bounds
        return (xy[..., 0] >= xmin) & (xy[..., 0] <= xmax) & (xy[..., 1] >= ymin) & (xy[..., 1] <= ymax)

    def validate(self):
        points = np.asarray(self.waypoints, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2 or not np.isfinite(points).all():
            raise ValueError("The route needs at least two finite XY waypoints")
        if not np.isfinite(self.reach_radius) or self.reach_radius <= 0:
            raise ValueError("reach_radius must be positive")
        if np.any(np.linalg.norm(np.diff(points, axis=0), axis=1) <= 2 * self.reach_radius):
            raise ValueError("Consecutive acceptance circles must not overlap")
        if not np.isfinite(self.spawn_jitter) or self.spawn_jitter < 0:
            raise ValueError("spawn_jitter must be nonnegative")
        bounds = np.asarray(self.bounds)
        if not np.isfinite(bounds).all() or bounds[0] >= bounds[1] or bounds[2] >= bounds[3]:
            raise ValueError("bounds must be (xmin, xmax, ymin, ymax)")
        if not np.all(self.contains(points)) or not self.contains(np.asarray(self.spawn_xy)):
            raise ValueError("Spawn and route must be inside the task bounds")
        if self.spawn_yaw_range[0] > self.spawn_yaw_range[1] or not np.isfinite(self.spawn_yaw_range).all():
            raise ValueError("spawn_yaw_range must be finite and ordered")


@configclass
class WaypointTaskCfg:
    hold_seconds: float = 3.0
    goal_radius: float = 0.35
    stop_radius: float = 0.25
    linear_kp: float = 0.8
    yaw_kp: float = 1.0
    max_forward: float = 0.6
    max_lateral: float = 0.15
    max_yaw_rate: float = 0.8
    slow_radius: float = 0.8
    turn_in_place_angle: float = 1.0
    standing_max_xy_speed: float = 0.10
    standing_max_z_speed: float = 0.10
    standing_max_angular_speed: float = 0.25
    standing_min_up_z: float = 0.97
    standing_height_tolerance: float = 0.08
    foot_height_tolerance: float = 0.065
    minimum_contacts: int = 3
    fall_min_base_height: float = 0.12
    progress_scale: float = 4.0
    waypoint_bonus: float = 5.0
    standing_scale: float = 3.0
    tilt_cost: float = 1.0
    success_bonus: float = 30.0
    failure_penalty: float = 2.0

    def validate(self, ctrl_dt, route):
        positive = (
            self.hold_seconds,
            self.goal_radius,
            self.stop_radius,
            self.linear_kp,
            self.yaw_kp,
            self.max_forward,
            self.max_lateral,
            self.max_yaw_rate,
            self.slow_radius,
            self.turn_in_place_angle,
            self.standing_max_xy_speed,
            self.standing_max_z_speed,
            self.standing_max_angular_speed,
            self.standing_height_tolerance,
            self.foot_height_tolerance,
            self.fall_min_base_height,
        )
        if not np.isfinite(positive).all() or min(positive) <= 0:
            raise ValueError("Task durations, gains, distances and speed limits must be positive")
        if not self.stop_radius < min(self.goal_radius, route.reach_radius, self.slow_radius):
            raise ValueError("Stop radius must be inside both acceptance circles and slow radius")
        if self.hold_seconds < ctrl_dt or not 0 < self.standing_min_up_z <= 1:
            raise ValueError("Invalid hold interval or upright threshold")
        if self.minimum_contacts not in (3, 4):
            raise ValueError("Standing requires three or four foot contacts")


@registry.envcfg("go1-waypoint-flat")
@configclass
class Go1WaypointFlatEnvCfg(Go1WalkDirectEnvCfg):
    route: WaypointRouteCfg = WaypointRouteCfg()
    task: WaypointTaskCfg = WaypointTaskCfg()
    max_episode_seconds: float = 40.0
    initial_base_position: tuple[float, float, float] = (-3.0, -2.0, 0.3)
    spawn_xy_range: float = 0.0
    noise_config: NoiseConfig = NoiseConfig(level=0.0)
    sim: SimCfg = SimCfg(dt=0.01, solver_iterations=4)

    def __post_init__(self):
        self.scene = QuadrupedSceneCfg(
            file=ROOT / "assets/route_markers.xml",
            objs=StandardSceneObjsCfg(
                floor=FlatTerrainCfg(material="mat_ground", friction=(0.6, 0.005, 0.0001)),
                robot=UnitreeGo1Robot(),
            ),
            system_camera=SystemCameraCfg(
                lookat=(-0.5, -0.5, 0.15),
                distance=10.0,
                elevation=-50.0,
                azimuth=90.0,
            ),
        )

    def validate(self):
        super().validate()
        self.route.validate()
        self.task.validate(self.ctrl_dt, self.route)
        if self.spawn_xy_range != 0:
            raise ValueError("The course uses only the route's configured spawn rectangle")
        if self.max_episode_seconds <= self.task.hold_seconds:
            raise ValueError("Leave time to walk the route before the final hold")

    def for_play(self):
        return replace(
            self,
            route=replace(self.route, spawn_jitter=0.0, spawn_yaw_range=(0.0, 0.0)),
            randomization=replace(self.randomization, enabled=False),
            noise_config=replace(self.noise_config, level=0.0),
        )
