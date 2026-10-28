# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Pure NumPy navigation helpers; observations must never advance this state.

The environment must settle progress and arrival rewards against its old target,
then call ``advance_waypoints`` once, before producing the next observation.
The helpers neither reset a robot nor loop a completed route back to its start.
"""

from __future__ import annotations

import math

import numpy as np

DEFAULT_WAYPOINTS = ((-1.0, -2.0), (-1.0, 1.0), (2.0, 1.0))


def wrap_angle(angle: np.ndarray) -> np.ndarray:
    """Return signed shortest angular differences in [-pi, pi)."""
    return (np.asarray(angle) + np.pi) % (2.0 * np.pi) - np.pi


def command_from_target(
    base_xy: np.ndarray,
    yaw: np.ndarray,
    target_xy: np.ndarray,
    *,
    linear_kp: float = 0.8,
    yaw_kp: float = 1.0,
    max_forward: float = 0.6,
    max_lateral: float = 0.2,
    max_yaw_rate: float = 1.0,
    slow_radius: float = 0.8,
    stop_radius: float = 0.25,
    turn_in_place_angle: float = 1.0,
) -> np.ndarray:
    """Make body-frame [vx, vy, yaw rate] commands for each current target.

    Large heading errors first command a turn in place. Translation gradually
    returns as the robot faces its target, and decreases near the target. All
    commands, including yaw, are zero within ``stop_radius``. Intermediate
    arrival should switch to the next target before this smaller stop radius.
    """
    base_xy = np.asarray(base_xy)
    target_xy = np.asarray(target_xy)
    yaw = np.asarray(yaw)
    if base_xy.ndim != 2 or base_xy.shape[1] != 2:
        raise ValueError("base_xy must have shape (num_envs, 2)")
    if target_xy.shape != base_xy.shape or yaw.shape != (len(base_xy),):
        raise ValueError("target_xy and yaw must match the base batch")
    limits = (linear_kp, yaw_kp, max_forward, max_lateral, max_yaw_rate, slow_radius, turn_in_place_angle)
    if not np.all(np.isfinite(limits)) or min(limits) <= 0:
        raise ValueError("Navigation gains, limits and radii must be positive")
    if not np.isfinite(stop_radius) or not 0 <= stop_radius < slow_radius:
        raise ValueError("stop_radius must be nonnegative and below slow_radius")

    delta = target_xy - base_xy
    distance = np.linalg.norm(delta, axis=1)
    cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
    body_x = cos_yaw * delta[:, 0] + sin_yaw * delta[:, 1]
    body_y = -sin_yaw * delta[:, 0] + cos_yaw * delta[:, 1]
    heading_error = wrap_angle(np.arctan2(delta[:, 1], delta[:, 0]) - yaw)
    # A bounded heading factor avoids strafe-only completion of sharp turns.
    heading_factor = np.clip(1.0 - np.abs(heading_error) / turn_in_place_angle, 0.0, 1.0)
    approach_factor = np.clip(distance / slow_radius, 0.0, 1.0)
    translation_factor = heading_factor * approach_factor
    commands = np.stack(
        (
            np.clip(linear_kp * body_x, -max_forward, max_forward) * translation_factor,
            np.clip(linear_kp * body_y, -max_lateral, max_lateral) * translation_factor,
            np.clip(yaw_kp * heading_error, -max_yaw_rate, max_yaw_rate),
        ),
        axis=1,
    )
    commands[distance <= stop_radius] = 0.0
    return commands.astype(np.float32, copy=False)


def advance_waypoints(
    indices: np.ndarray,
    base_xy: np.ndarray,
    waypoints: np.ndarray,
    radius: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Advance each reached intermediate waypoint by exactly one index.

    A final target remains final. Calling this exactly once per control step
    prevents coincident/overlapping targets from being skipped in one step.
    No input arrays are modified. The returned mask is a one-step arrival event.
    """
    indices = np.asarray(indices)
    base_xy = np.asarray(base_xy)
    waypoints = np.asarray(waypoints)
    if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer):
        raise ValueError("indices must be a one-dimensional integer batch")
    if base_xy.shape != (len(indices), 2):
        raise ValueError("base_xy must have shape (num_envs, 2)")
    if waypoints.ndim != 2 or waypoints.shape[1] != 2 or len(waypoints) < 1:
        raise ValueError("waypoints must contain at least one XY target")
    if np.any(indices < 0) or np.any(indices >= len(waypoints)):
        raise ValueError("A waypoint index is outside the finite route")
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("Waypoint arrival radius must be positive")
    distance = np.linalg.norm(base_xy - waypoints[indices], axis=1)
    reached_intermediate = (distance <= radius) & (indices < len(waypoints) - 1)
    return indices + reached_intermediate.astype(indices.dtype), reached_intermediate


def standing_conditions(
    task,
    target_height: float,
    *,
    base: np.ndarray,
    ground: np.ndarray,
    feet: np.ndarray,
    contacts: np.ndarray,
    velocity: np.ndarray,
    gyro: np.ndarray,
    up: np.ndarray,
    target_xy: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Measure flat-ground arrival and support; this does not advance a clock.

    The caller must additionally restrict ``stable`` to the route's final
    index. ``velocity`` may use world or body axes: its horizontal norm and
    vertical absolute value are yaw invariant. Feet and base are world poses.
    """
    ground = np.broadcast_to(np.asarray(ground), (len(base),))
    contacts = np.asarray(contacts, dtype=bool)
    if np.asarray(feet).shape != (len(base), 4, 3) or contacts.shape != (len(base), 4):
        raise ValueError("Standing checks require four world-frame feet and contacts")
    distance = np.linalg.norm(np.asarray(base)[:, :2] - target_xy, axis=1)
    arrived = distance <= task.goal_radius
    contacting_feet_at_ground = np.all(
        ~contacts | (np.abs(feet[:, :, 2] - ground[:, None]) <= task.foot_height_tolerance),
        axis=1,
    )
    stable = (
        arrived
        & (contacts.sum(axis=1) >= task.minimum_contacts)
        & contacting_feet_at_ground
        & (up[:, 2] >= task.standing_min_up_z)
        & (np.abs(base[:, 2] - ground - target_height) <= task.standing_height_tolerance)
        & (np.linalg.norm(velocity[:, :2], axis=1) <= task.standing_max_xy_speed)
        & (np.abs(velocity[:, 2]) <= task.standing_max_z_speed)
        & (np.linalg.norm(gyro, axis=1) <= task.standing_max_angular_speed)
    )
    return arrived, stable


class StandingClock:
    """Count consecutive stable control steps separately for each environment."""

    def __init__(self, num_envs: int, hold_seconds: float, ctrl_dt: float):
        if num_envs < 1 or not isinstance(num_envs, int):
            raise ValueError("num_envs must be a positive integer")
        if not math.isfinite(hold_seconds) or not math.isfinite(ctrl_dt):
            raise ValueError("Standing duration and control timestep must be finite")
        if hold_seconds <= 0 or ctrl_dt <= 0:
            raise ValueError("Standing duration and control timestep must be positive")
        self.ctrl_dt = ctrl_dt
        self.required_steps = math.ceil(hold_seconds / ctrl_dt - 1e-10)
        self.steps = np.zeros(num_envs, dtype=np.int64)

    @property
    def seconds(self) -> np.ndarray:
        return self.steps * self.ctrl_dt

    def update(self, stable: np.ndarray, failed: np.ndarray) -> np.ndarray:
        stable = np.asarray(stable, dtype=bool)
        failed = np.asarray(failed, dtype=bool)
        if stable.shape != self.steps.shape or failed.shape != self.steps.shape:
            raise ValueError("Standing and failure flags must match the environment batch")
        self.steps[:] = np.where(
            stable & ~failed,
            np.minimum(self.steps + 1, self.required_steps),
            0,
        )
        return (self.steps >= self.required_steps) & ~failed

    def reset(self, env_ids: np.ndarray) -> None:
        self.steps[env_ids] = 0
