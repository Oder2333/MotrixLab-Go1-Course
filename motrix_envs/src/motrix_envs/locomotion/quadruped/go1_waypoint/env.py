# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Finite sequential waypoints; a rules controller commands a learned gait."""

import gymnasium as gym
import numpy as np

from motrix_env_core import registry
from motrix_env_core.array.env import NpObs
from motrix_env_core.base import ObsSpace
from motrix_env_core.sim.read import LinkQuaternionQuery, SitePositionQuery
from motrix_envs.locomotion.quadruped.walk_np import QuadrupedWalkTask

from .cfg import Go1WaypointFlatEnvCfg
from .logic import StandingClock, advance_waypoints, command_from_target, standing_conditions


@registry.env("go1-waypoint-flat")
class Go1WaypointFlatTask(QuadrupedWalkTask):
    def __init__(self, cfg: Go1WaypointFlatEnvCfg, num_envs=1, backend=None):
        super().__init__(cfg, num_envs, backend=backend)
        self._env_ids = np.arange(num_envs, dtype=np.int64)
        self._waypoints = np.asarray(cfg.route.waypoints, dtype=np.float32)
        self._route_index = np.zeros(num_envs, dtype=np.int64)
        self._goal_body = np.zeros((num_envs, 2), dtype=np.float32)
        self._goal_distance = np.zeros(num_envs, dtype=np.float32)
        self._last_distance = np.zeros(num_envs, dtype=np.float32)
        self._yaw = np.zeros(num_envs, dtype=np.float32)
        self._feet_world = np.zeros((num_envs, 4, 3), dtype=np.float32)
        robot = cfg.scene.objs.robot
        self._extra_reads = self.sim.compile_reads(
            {
                "base_quat": LinkQuaternionQuery(link=self._base_link_name),
                **{
                    f"foot_{i}": SitePositionQuery(site=robot.resolve_name(name))
                    for i, name in enumerate(("FL", "FR", "RL", "RR"))
                },
            }
        )
        self._clock = StandingClock(num_envs, cfg.task.hold_seconds, cfg.ctrl_dt)
        self._episode_steps = np.zeros(num_envs, dtype=np.int64)
        self._episode_return = np.zeros(num_envs, dtype=np.float32)
        self._has_episode = np.zeros(num_envs, dtype=bool)
        self._success = np.zeros(num_envs, dtype=bool)
        self._fallen = np.zeros(num_envs, dtype=bool)
        self._boundary = np.zeros(num_envs, dtype=bool)
        self._timeout = np.zeros(num_envs, dtype=bool)
        self._arrived = np.zeros(num_envs, dtype=bool)
        self.episodic_stats = {}
        self.completed_episodes = 0
        self.successful_episodes = 0

    def _init_obs_space(self):
        self._observation_space = ObsSpace(
            policy=gym.spaces.Box(-np.inf, np.inf, (self._policy_obs_dim() + 3,), np.float32),
            value=gym.spaces.Box(-np.inf, np.inf, (self._value_obs_dim() + 5,), np.float32),
        )

    def apply_action(self, actions, state):
        if not np.isfinite(actions).all():
            raise ValueError("Policy actions must be finite")
        bounded = np.clip(actions, self.action_space.low, self.action_space.high)
        return super().apply_action(bounded.astype(np.float32), state)

    def _refresh_goal(self, env_ids):
        base = self.sim_data["base_pos"][env_ids]
        delta = self._waypoints[self._route_index[env_ids]] - base[:, :2]
        yaw = self._yaw[env_ids]
        c, s = np.cos(yaw), np.sin(yaw)
        self._goal_body[env_ids] = np.column_stack(
            (c * delta[:, 0] + s * delta[:, 1], -s * delta[:, 0] + c * delta[:, 1])
        )
        self._goal_distance[env_ids] = np.linalg.norm(delta, axis=1)

    def _refresh_task_geometry(self, env_ids):
        self._extra_reads.execute(env_ids)
        x, y, z, w = self._extra_reads["base_quat"][env_ids].T
        self._yaw[env_ids] = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        for i in range(4):
            self._feet_world[env_ids, i] = self._extra_reads[f"foot_{i}"][env_ids]
        self._refresh_goal(env_ids)

    def _update_goal_commands(self, env_ids):
        task = self.cfg.task
        self._commands[env_ids] = command_from_target(
            self.sim_data["base_pos"][env_ids, :2],
            self._yaw[env_ids],
            self._waypoints[self._route_index[env_ids]],
            linear_kp=task.linear_kp,
            yaw_kp=task.yaw_kp,
            max_forward=task.max_forward,
            max_lateral=task.max_lateral,
            max_yaw_rate=task.max_yaw_rate,
            slow_radius=task.slow_radius,
            stop_radius=task.stop_radius,
            turn_in_place_angle=task.turn_in_place_angle,
        )
        stopped = env_ids[
            np.linalg.norm(self._commands[env_ids], axis=1) < self.cfg.commands.velocity.standing_threshold
        ]
        self._phase[stopped] = 0.0
        self._feet_phase[stopped] = 0.0

    def standing_mask(self):
        arrived, stable = standing_conditions(
            self.cfg.task,
            self.cfg.reward_config.base_height_target,
            base=self.sim_data["base_pos"],
            ground=0.0,
            feet=self._feet_world,
            contacts=self.feet_contact,
            velocity=self.get_local_linvel(),
            gyro=self.get_gyro(),
            up=self.get_gravity(),
            target_xy=self._waypoints[-1],
        )
        final = self._route_index == len(self._waypoints) - 1
        return arrived & final, stable & final

    def compute_transition(self, state):
        self.episodic_stats = {}
        self.sim_data.execute()
        self._refresh_task_geometry(self._env_ids)
        self._advance_phase()
        self._update_feet_buffers()
        base, task = self.sim_data["base_pos"], self.cfg.task
        invalid = ~np.isfinite(
            np.concatenate(
                (
                    base,
                    self.get_local_linvel(),
                    self.get_gyro(),
                    self.get_gravity(),
                    self.get_dof_pos(),
                    self.get_dof_vel(),
                    self._feet_world.reshape(self.num_envs, -1),
                ),
                axis=1,
            )
        ).all(axis=1)
        self._fallen[:] = (
            (self.get_gravity()[:, 2] <= self.cfg.termination_min_up_z)
            | (base[:, 2] < task.fall_min_base_height)
            | invalid
        )
        self._boundary[:] = ~self.cfg.route.contains(base[:, :2])
        failed = self._fallen | self._boundary
        arrived, stable = self.standing_mask()
        stable &= ~failed
        self._arrived[:] = arrived & ~failed
        self._success[:] = self._clock.update(stable, failed)
        self._episode_steps += 1
        self._timeout[:] = (self._episode_steps >= self.cfg.max_episode_steps) & ~failed & ~self._success

        # Score the action against exactly the command and goal it observed.
        reward = self._compute_reward(state, self.get_local_linvel(), self.get_gyro(), self.get_dof_pos())
        progress = np.clip((self._last_distance - self._goal_distance) / self.cfg.ctrl_dt, -1.0, 1.0)
        next_index, reached = advance_waypoints(
            self._route_index, base[:, :2], self._waypoints, self.cfg.route.reach_radius
        )
        reached &= ~failed & ~self._timeout
        extra_terms = {
            "goal_progress": task.progress_scale * progress * ~failed,
            "waypoint_reached": task.waypoint_bonus / self.cfg.ctrl_dt * reached,
            "final_standing": task.standing_scale * stable,
            "body_tilt": -task.tilt_cost * np.square(self.get_gravity()[:, :2]).sum(axis=1),
            "task_success": task.success_bonus / self.cfg.ctrl_dt * self._success,
            "task_failure": -task.failure_penalty / self.cfg.ctrl_dt * failed,
        }
        for name, term in extra_terms.items():
            term = np.asarray(term, dtype=np.float32)
            state.reward_terms[name] = term
            reward += term * self.cfg.ctrl_dt
        if np.any(invalid):
            reward[invalid] = -task.failure_penalty
            for name, term in state.reward_terms.items():
                if name != "task_failure":
                    term[invalid] = 0.0
        self._episode_return += reward
        self._route_index[reached] = next_index[reached]
        self._refresh_goal(self._env_ids)
        self._last_distance[:] = self._goal_distance  # Rebase after a switch.
        self._update_goal_commands(self._env_ids)
        # Owned arrays retain the completed route outcome through auto-reset.
        state.metrics = {
            "task/success": self._success.copy(),
            "task/fallen": self._fallen.copy(),
            "task/nonfinite": invalid.copy(),
            "task/boundary": self._boundary.copy(),
            "task/arrived": self._arrived.copy(),
            "task/waypoint_reached": reached.copy(),
            "task/waypoints_completed": (self._route_index + self._success).copy(),
            "task/current_waypoint": (self._route_index + 1).copy(),
            "task/standing_seconds": self._clock.seconds.copy(),
            "task/goal_distance": np.nan_to_num(self._goal_distance.copy()),
        }
        return state.replace(reward=reward, terminated=failed | self._success)

    def compute_observation(self, state):
        base = super().compute_observation(state)
        route_fraction = self._route_index / (len(self._waypoints) - 1)
        extra_value = np.column_stack(
            (
                np.clip(self._goal_body / 10.0, -1.0, 1.0),
                route_fraction,
                self._route_index == len(self._waypoints) - 1,
                self._clock.steps / self._clock.required_steps,
            )
        ).astype(np.float32)
        return base.replace(
            obs=NpObs(
                policy=np.concatenate((base.obs.policy, self.get_local_linvel()), axis=1),
                value=np.concatenate((base.obs.value, extra_value), axis=1),
            )
        )

    def reset(self, env_ids):
        completed = self._success | self._fallen | self._boundary | self._timeout
        finished = env_ids[self._has_episode[env_ids] & (self._episode_steps[env_ids] > 0) & completed[env_ids]]
        if finished.size:
            self.completed_episodes += len(finished)
            self.successful_episodes += int(self._success[finished].sum())
            self.episodic_stats = {
                "success/Episode": float(np.mean(self._success[finished])),
                "fallen/Episode": float(np.mean(self._fallen[finished])),
                "boundary/Episode": float(np.mean(self._boundary[finished])),
                "timeout/Episode": float(np.mean(self._timeout[finished])),
                "waypoints_completed/Episode": float(np.mean(self._route_index[finished] + self._success[finished])),
                "length_seconds/Episode": float(np.mean(self._episode_steps[finished]) * self.cfg.ctrl_dt),
                "return/Episode": float(np.mean(self._episode_return[finished])),
            }
        super().reset(env_ids)
        route = self.cfg.route
        count = len(env_ids)
        self._reset_position[env_ids, :2] = np.asarray(route.spawn_xy) + self._randomization_rng.uniform(
            -route.spawn_jitter, route.spawn_jitter, (count, 2)
        )
        yaw = self._randomization_rng.uniform(*route.spawn_yaw_range, count)
        self._reset_rotation[env_ids] = np.column_stack(
            (np.zeros(count), np.zeros(count), np.sin(yaw / 2), np.cos(yaw / 2))
        )
        self._reset_program.execute(env_ids)
        self.sim_data.execute(env_ids)
        self._update_feet_buffers(env_ids)
        self._route_index[env_ids] = 0
        self._clock.reset(env_ids)
        self._episode_steps[env_ids] = 0
        self._episode_return[env_ids] = 0
        self._success[env_ids] = self._fallen[env_ids] = self._boundary[env_ids] = False
        self._timeout[env_ids] = self._arrived[env_ids] = False
        self._has_episode[env_ids] = True
        self._refresh_task_geometry(env_ids)
        self._last_distance[env_ids] = self._goal_distance[env_ids]
        self._update_goal_commands(env_ids)
