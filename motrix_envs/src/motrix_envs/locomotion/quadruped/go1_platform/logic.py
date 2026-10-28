# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Simulator-independent, consecutive control-step standing clock."""

import math

import numpy as np


def standing_conditions(task, geometry, target_height, *, base, ground, feet, contacts, velocity, gyro, up):
    """Evaluate measured support and stability, without advancing time."""
    on_platform = np.all(
        (feet[:, :, 0] >= geometry.ramp_end_x + task.platform_foot_margin)
        & geometry.contains(feet[:, :, 0], feet[:, :, 1], margin=geometry.boundary_margin)
        & (np.abs(feet[:, :, 2] - geometry.platform_height) <= task.foot_height_tolerance),
        axis=1,
    )
    distance = np.linalg.norm(base[:, :2] - np.asarray(geometry.goal_xy), axis=1)
    arrived = (distance <= task.goal_radius) & on_platform
    stable = (
        arrived
        & (contacts.sum(axis=1) >= task.minimum_contacts)
        & (up[:, 2] >= task.standing_min_up_z)
        & (np.abs(base[:, 2] - ground - target_height) <= task.standing_height_tolerance)
        & (np.linalg.norm(velocity[:, :2], axis=1) <= task.standing_max_xy_speed)
        & (np.abs(velocity[:, 2]) <= task.standing_max_z_speed)
        & (np.linalg.norm(gyro, axis=1) <= task.standing_max_angular_speed)
    )
    return arrived, stable


class StandingClock:
    def __init__(self, num_envs: int, hold_seconds: float, ctrl_dt: float):
        self.ctrl_dt = ctrl_dt
        self.required_steps = math.ceil(hold_seconds / ctrl_dt - 1e-10)
        self.steps = np.zeros(num_envs, dtype=np.int64)

    @property
    def seconds(self) -> np.ndarray:
        return self.steps * self.ctrl_dt

    def update(self, stable: np.ndarray, failed: np.ndarray) -> np.ndarray:
        self.steps[:] = np.where(stable & ~failed, self.steps + 1, 0)
        return (self.steps >= self.required_steps) & ~failed

    def reset(self, env_ids: np.ndarray) -> None:
        self.steps[env_ids] = 0
