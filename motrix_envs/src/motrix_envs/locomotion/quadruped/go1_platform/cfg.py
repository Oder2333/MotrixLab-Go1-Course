# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Lesson 2: shared geometry, goal commands and standing acceptance."""

from dataclasses import replace

import numpy as np
from motrix_robots.unitree import UnitreeGo1Robot

from motrix_env_core import registry
from motrix_env_core.base import SimCfg
from motrix_env_core.config import configclass
from motrix_env_core.config.scene import HFieldTerrainCfg
from motrix_env_core.config.scene.asset import ProceduralHFieldAssetCfg
from motrix_env_core.config.scene.base import SystemCameraCfg
from motrix_envs.config.scene import StandardSceneAssetsCfg, StandardSceneObjsCfg
from motrix_envs.locomotion.quadruped.cfg import NoiseConfig, QuadrupedSceneCfg
from motrix_envs.locomotion.quadruped.go1 import Go1WalkDirectEnvCfg

from .terrain import DEFAULT_GEOMETRY, SlopePlatformGeometryCfg, make_terrain_asset


@configclass
class PlatformTaskCfg:
    hold_seconds: float = 3.0
    goal_radius: float = 0.35
    stop_radius: float = 0.25
    max_speed: float = 0.5
    position_gain: float = 0.8
    max_yaw_rate: float = 0.6
    standing_max_xy_speed: float = 0.10
    standing_max_z_speed: float = 0.10
    standing_max_angular_speed: float = 0.25
    standing_min_up_z: float = 0.97
    standing_height_tolerance: float = 0.08
    foot_height_tolerance: float = 0.065
    platform_foot_margin: float = 0.15
    minimum_contacts: int = 3
    fall_min_base_height: float = 0.12
    progress_scale: float = 4.0
    goal_scale: float = 2.0
    standing_scale: float = 3.0
    lateral_cost: float = 0.5
    tilt_cost: float = 1.0
    success_bonus: float = 30.0
    failure_penalty: float = 2.0

    def validate(self, ctrl_dt: float, geometry: SlopePlatformGeometryCfg) -> None:
        positive = (
            self.hold_seconds,
            self.goal_radius,
            self.stop_radius,
            self.max_speed,
            self.position_gain,
            self.max_yaw_rate,
            self.standing_max_xy_speed,
            self.standing_max_z_speed,
            self.standing_max_angular_speed,
            self.standing_height_tolerance,
            self.foot_height_tolerance,
            self.platform_foot_margin,
            self.fall_min_base_height,
        )
        if not np.all(np.isfinite(positive)) or min(positive) <= 0:
            raise ValueError("Task distances, speeds and durations must be finite and positive")
        if not self.stop_radius < self.goal_radius:
            raise ValueError("stop_radius must be smaller than the acceptance goal_radius")
        if not 0 < self.standing_min_up_z <= 1 or self.minimum_contacts not in (3, 4):
            raise ValueError("Standing needs a valid upright threshold and 3 or 4 contacts")
        if self.hold_seconds < ctrl_dt:
            raise ValueError("The hold interval must span at least one control step")
        if geometry.goal_xy[0] - self.goal_radius < geometry.ramp_end_x + self.platform_foot_margin:
            raise ValueError("Goal acceptance region must be clear of the ramp")


@configclass
class PlatformSceneAssetsCfg(StandardSceneAssetsCfg):
    terrain: ProceduralHFieldAssetCfg = make_terrain_asset()


@registry.envcfg("go1-slope-platform")
@configclass
class Go1SlopePlatformEnvCfg(Go1WalkDirectEnvCfg):
    """Walk from the flat approach to a platform goal, then stand for 3 s."""

    geometry: SlopePlatformGeometryCfg = DEFAULT_GEOMETRY
    platform_task: PlatformTaskCfg = PlatformTaskCfg()
    max_episode_seconds: float = 30.0
    spawn_xy_range: float = 0.0
    initial_base_position: tuple[float, float, float] = (-4.25, 0.0, 0.3)
    noise_config: NoiseConfig = NoiseConfig(level=0.0)
    sim: SimCfg = SimCfg(dt=0.01, solver_iterations=4)

    def __post_init__(self) -> None:
        self.scene = QuadrupedSceneCfg(
            assets=PlatformSceneAssetsCfg(terrain=make_terrain_asset(self.geometry)),
            objs=StandardSceneObjsCfg(
                floor=HFieldTerrainCfg(hfield="terrain", material="mat_ground", friction=(0.6, 0.005, 0.0001)),
                robot=UnitreeGo1Robot(),
            ),
            system_camera=SystemCameraCfg(lookat=(-0.5, 0.0, 0.25), distance=10.0, elevation=-25.0, azimuth=90.0),
        )

    def validate(self) -> None:
        super().validate()
        self.geometry.validate()
        self.platform_task.validate(self.ctrl_dt, self.geometry)
        if self.spawn_xy_range != 0:
            raise ValueError("Lesson 2 uses only the configured flat spawn rectangle")
        if self.max_episode_seconds <= self.platform_task.hold_seconds:
            raise ValueError("The episode must leave time to approach the platform")

    def for_play(self) -> "Go1SlopePlatformEnvCfg":
        return replace(
            self,
            randomization=replace(self.randomization, enabled=False),
            noise_config=replace(self.noise_config, level=0.0),
        )
