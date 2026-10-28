# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Goal-conditioned Go1 task using the pinned native MotrixSim backend."""

import gymnasium as gym
import numpy as np

from motrix_env_core import registry
from motrix_env_core.array.env import ArrayEnvState, NpObs
from motrix_env_core.base import ObsSpace
from motrix_env_core.sim.read import LinkQuaternionQuery, SitePositionQuery
from motrix_env_core.sim.write import BodyPositionWrite
from motrix_envs.locomotion.quadruped.walk_np import QuadrupedWalkTask

from .cfg import Go1SlopePlatformEnvCfg
from .logic import StandingClock, standing_conditions

_SCAN_X, _SCAN_Y = np.meshgrid(
    np.array([-0.5, 0.0, 0.5, 1.0, 1.5], dtype=np.float32),
    np.array([-0.3, 0.0, 0.3], dtype=np.float32),
    indexing="ij",
)
_SCAN_POINTS = np.stack((_SCAN_X, _SCAN_Y), axis=-1).reshape(-1, 2)
# goal XY (2), goal height (1), hold fraction (1), local velocity (3), scan (15).
_EXTRA_DIM = 7 + len(_SCAN_POINTS)


@registry.env("go1-slope-platform")
class Go1SlopePlatformTask(QuadrupedWalkTask):
    def __init__(self, cfg: Go1SlopePlatformEnvCfg, num_envs=1, backend=None):
        super().__init__(cfg, num_envs, backend=backend)
        self._env_ids = np.arange(num_envs, dtype=np.int64)
        robot = cfg.scene.objs.robot
        # Parent foot sensors are relative to the offset IMU site, not the base
        # origin. Read world foot sites directly for platform acceptance.
        self._extra_reads = self.sim.compile_reads(
            {
                "base_quat": LinkQuaternionQuery(link=self._base_link_name),
                **{
                    f"foot_{i}": SitePositionQuery(site=robot.resolve_name(name))
                    for i, name in enumerate(("FL", "FR", "RL", "RR"))
                },
            }
        )
        self._spawn_program = self.sim.write_compiler.compile(
            {"base_position": BodyPositionWrite((self._base_link_name,))},
            reset=False,
            forward_kinematics=True,
        )
        self._clock = StandingClock(num_envs, cfg.platform_task.hold_seconds, cfg.ctrl_dt)
        self._goal = np.asarray(cfg.geometry.goal_xy, dtype=np.float32)
        self._ground_height = np.zeros(num_envs, dtype=np.float32)
        self._goal_distance = np.zeros(num_envs, dtype=np.float32)
        self._last_distance = np.zeros(num_envs, dtype=np.float32)
        self._goal_body = np.zeros((num_envs, 2), dtype=np.float32)
        self._height_scan = np.zeros((num_envs, len(_SCAN_POINTS)), dtype=np.float32)
        self._feet_world = np.zeros((num_envs, 4, 3), dtype=np.float32)
        self._yaw = np.zeros(num_envs, dtype=np.float32)
        self._episode_steps = np.zeros(num_envs, dtype=np.int64)
        self._episode_return = np.zeros(num_envs, dtype=np.float32)
        self._has_episode = np.zeros(num_envs, dtype=bool)
        self._success = np.zeros(num_envs, dtype=bool)
        self._fallen = np.zeros(num_envs, dtype=bool)
        self._boundary = np.zeros(num_envs, dtype=bool)
        self._timeout = np.zeros(num_envs, dtype=bool)
        self._arrived = np.zeros(num_envs, dtype=bool)
        self.episodic_stats: dict[str, float] = {}
        self.completed_episodes = 0
        self.successful_episodes = 0

    def _init_obs_space(self) -> None:
        # Keep parent's dimension methods unchanged for its own observation
        # construction; append task observations only after that check passes.
        self._observation_space = ObsSpace(
            policy=gym.spaces.Box(-np.inf, np.inf, (self._policy_obs_dim() + _EXTRA_DIM,), np.float32),
            value=gym.spaces.Box(-np.inf, np.inf, (self._value_obs_dim() + _EXTRA_DIM,), np.float32),
        )

    def apply_action(self, actions, state):
        bounded = np.clip(actions, self.action_space.low, self.action_space.high)
        return super().apply_action(bounded.astype(np.float32), state)

    def _refresh_task_geometry(self, env_ids):
        self._extra_reads.execute(env_ids)
        base = self.sim_data["base_pos"][env_ids]
        x, y, z, w = self._extra_reads["base_quat"][env_ids].T
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        self._yaw[env_ids] = yaw
        c, s = np.cos(yaw), np.sin(yaw)
        delta = self._goal - base[:, :2]
        self._goal_distance[env_ids] = np.linalg.norm(delta, axis=1)
        self._goal_body[env_ids] = np.column_stack(
            (c * delta[:, 0] + s * delta[:, 1], -s * delta[:, 0] + c * delta[:, 1])
        )
        points = np.stack(
            (
                c[:, None] * _SCAN_POINTS[:, 0] - s[:, None] * _SCAN_POINTS[:, 1],
                s[:, None] * _SCAN_POINTS[:, 0] + c[:, None] * _SCAN_POINTS[:, 1],
            ),
            axis=-1,
        )
        points += base[:, None, :2]
        ground = self.sim.sample_terrain_height(self.cfg.ground_geom_name, env_ids, base[:, None, :2])[:, 0]
        scan = self.sim.sample_terrain_height(self.cfg.ground_geom_name, env_ids, points)
        self._ground_height[env_ids] = ground
        self._height_scan[env_ids] = np.clip(scan - ground[:, None], -1.0, 1.0)
        for i in range(4):
            self._feet_world[env_ids, i] = self._extra_reads[f"foot_{i}"][env_ids]

    def _update_goal_commands(self, env_ids):
        task = self.cfg.platform_task
        velocity = self._goal_body[env_ids] * task.position_gain
        speed = np.linalg.norm(velocity, axis=1)
        velocity *= np.minimum(1.0, task.max_speed / np.maximum(speed, 1e-6))[:, None]
        self._commands[env_ids, :2] = velocity
        self._commands[env_ids, 2] = np.clip(-1.5 * self._yaw[env_ids], -task.max_yaw_rate, task.max_yaw_rate)
        stopped = env_ids[self._goal_distance[env_ids] <= task.stop_radius]
        self._commands[stopped] = 0.0
        self._phase[stopped] = 0.0
        self._feet_phase[stopped] = 0.0

    def standing_mask(self):
        """Read-only acceptance check using refreshed world-space measurements."""
        return standing_conditions(
            self.cfg.platform_task,
            self.cfg.geometry,
            self.cfg.reward_config.base_height_target,
            base=self.sim_data["base_pos"],
            ground=self._ground_height,
            feet=self._feet_world,
            contacts=self.feet_contact,
            velocity=self.get_local_linvel(),
            gyro=self.get_gyro(),
            up=self.get_gravity(),
        )

    def compute_transition(self, state: ArrayEnvState) -> ArrayEnvState:
        self.episodic_stats = {}
        self.sim_data.execute()
        self._refresh_task_geometry(self._env_ids)
        self._advance_phase()
        self._update_feet_buffers()
        base = self.sim_data["base_pos"]
        geometry, task = self.cfg.geometry, self.cfg.platform_task
        self._fallen[:] = (self.get_gravity()[:, 2] <= self.cfg.termination_min_up_z) | (
            base[:, 2] - self._ground_height < task.fall_min_base_height
        )
        self._boundary[:] = ~geometry.contains(base[:, 0], base[:, 1], margin=geometry.boundary_margin)
        failed = self._fallen | self._boundary
        arrived, stable = self.standing_mask()
        self._arrived[:] = arrived
        self._success[:] = self._clock.update(stable, failed)
        self._episode_steps += 1
        self._timeout[:] = (self._episode_steps >= self.cfg.max_episode_steps) & ~failed & ~self._success

        # The action observed the previous command; score it before updating
        # the command that the next observation will expose.
        reward = self._compute_reward(state, self.get_local_linvel(), self.get_gyro(), self.get_dof_pos())
        progress = np.clip((self._last_distance - self._goal_distance) / self.cfg.ctrl_dt, -1.0, 1.0)
        extra_terms = {
            "goal_progress": task.progress_scale * progress,
            "goal_proximity": task.goal_scale * np.exp(-np.square(self._goal_distance / 0.5)),
            "platform_standing": task.standing_scale * stable.astype(np.float32),
            "route_lateral": -task.lateral_cost * np.square(base[:, 1] - geometry.goal_xy[1]),
            "body_tilt": -task.tilt_cost * np.sum(np.square(self.get_gravity()[:, :2]), axis=1),
            "task_success": task.success_bonus / self.cfg.ctrl_dt * self._success,
            "task_failure": -task.failure_penalty / self.cfg.ctrl_dt * failed,
        }
        for name, term in extra_terms.items():
            term = np.asarray(term, dtype=np.float32)
            state.reward_terms[name] = term
            reward += term * self.cfg.ctrl_dt
        self._episode_return += reward
        self._last_distance[:] = self._goal_distance
        self._update_goal_commands(self._env_ids)
        # Own snapshots survive the selected-row auto-reset that follows.
        state.metrics = {
            "task/success": self._success.copy(),
            "task/fallen": self._fallen.copy(),
            "task/boundary": self._boundary.copy(),
            "task/arrived": self._arrived.copy(),
            "task/standing_seconds": self._clock.seconds.copy(),
            "task/goal_distance": self._goal_distance.copy(),
        }
        return state.replace(reward=reward, terminated=failed | self._success)

    def compute_observation(self, state):
        base = super().compute_observation(state)
        goal_height = self.cfg.geometry.platform_height - self._ground_height
        extra = np.concatenate(
            (
                np.clip(self._goal_body / 6.0, -1.0, 1.0),
                goal_height[:, None],
                np.minimum(self._clock.steps / self._clock.required_steps, 1.0)[:, None],
                self.get_local_linvel(),
                self._height_scan,
            ),
            axis=1,
        ).astype(np.float32)
        return base.replace(
            obs=NpObs(
                policy=np.concatenate((base.obs.policy, extra), axis=1),
                value=np.concatenate((base.obs.value, extra), axis=1),
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
                "length_seconds/Episode": float(np.mean(self._episode_steps[finished]) * self.cfg.ctrl_dt),
                "return/Episode": float(np.mean(self._episode_return[finished])),
            }
        super().reset(env_ids)
        geometry = self.cfg.geometry
        position = self._spawn_program.buffer("base_position")[:, 0]
        position[env_ids, 0] = self._randomization_rng.uniform(*geometry.reset_x_range, len(env_ids))
        position[env_ids, 1] = self._randomization_rng.uniform(*geometry.reset_y_range, len(env_ids))
        position[env_ids, 2] = self.cfg.initial_base_position[2]
        self._spawn_program.execute(env_ids)
        self.sim_data.execute(env_ids)
        self._update_feet_buffers(env_ids)
        self._refresh_task_geometry(env_ids)
        self._last_distance[env_ids] = self._goal_distance[env_ids]
        self._clock.reset(env_ids)
        self._episode_steps[env_ids] = 0
        self._episode_return[env_ids] = 0
        self._success[env_ids] = self._fallen[env_ids] = self._boundary[env_ids] = False
        self._timeout[env_ids] = self._arrived[env_ids] = False
        self._has_episode[env_ids] = True
        self._update_goal_commands(env_ids)
