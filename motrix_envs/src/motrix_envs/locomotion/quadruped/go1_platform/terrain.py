# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Deterministic flat -> rounded ramp -> platform height field.

The only simulator-facing representation is the existing MotrixLab procedural
height-field contract: normalized float32 samples, C order, axis 0 = world Y,
axis 1 = world X. The backend applies ``height_scale`` exactly once. Coordinates
are local world coordinates with the terrain geometry at the origin.
"""

from __future__ import annotations

import numpy as np

from motrix_env_core.config import configclass
from motrix_env_core.config.scene.asset import ProceduralHFieldAssetCfg, TerrainGeneratorCfg


@configclass(frozen=True)
class SlopePlatformGeometryCfg:
    """Physical course dimensions, shared by scene construction and the task.

    The 0.25 m entrance/exit transitions have a linearly changing grade. They
    remove slope discontinuities without creating a step. The middle of the
    ramp has constant grade. There are no walls or implicit ground outside the
    finite field; ``boundary_margin`` defines the task's safe interior.
    """

    size: tuple[float, float] = (12.0, 6.0)
    # Native height-field rows follow Y; columns follow X.
    shape: tuple[int, int] = (121, 241)
    ramp_start_x: float = -3.0
    ramp_end_x: float = 0.0
    platform_height: float = 0.5
    transition_length: float = 0.25
    reset_x_range: tuple[float, float] = (-4.5, -4.0)
    reset_y_range: tuple[float, float] = (-0.25, 0.25)
    goal_xy: tuple[float, float] = (1.5, 0.0)
    boundary_margin: float = 0.5

    @property
    def x_bounds(self) -> tuple[float, float]:
        return (-self.size[0] / 2.0, self.size[0] / 2.0)

    @property
    def y_bounds(self) -> tuple[float, float]:
        return (-self.size[1] / 2.0, self.size[1] / 2.0)

    @property
    def ramp_length(self) -> float:
        return self.ramp_end_x - self.ramp_start_x

    @property
    def max_grade(self) -> float:
        return self.platform_height / (self.ramp_length - self.transition_length)

    def contains(self, x, y, *, margin: float = 0.0):
        """Return whether XY is inside the finite terrain, optionally inset."""
        x_array, y_array = np.broadcast_arrays(np.asarray(x), np.asarray(y))
        result = (
            (x_array >= self.x_bounds[0] + margin)
            & (x_array <= self.x_bounds[1] - margin)
            & (y_array >= self.y_bounds[0] + margin)
            & (y_array <= self.y_bounds[1] - margin)
        )
        return bool(result) if result.ndim == 0 else result

    def validate(self) -> None:
        values = (
            *self.size,
            self.ramp_start_x,
            self.ramp_end_x,
            self.platform_height,
            self.transition_length,
            *self.reset_x_range,
            *self.reset_y_range,
            *self.goal_xy,
            self.boundary_margin,
        )
        if not np.all(np.isfinite(values)):
            raise ValueError("Course geometry values must be finite")
        if any(extent <= 0 for extent in self.size):
            raise ValueError("Course size must be positive")
        if len(self.shape) != 2 or any(not isinstance(n, int) or n < 2 for n in self.shape):
            raise ValueError("Course shape must contain two integers >= 2")
        if not self.x_bounds[0] < self.ramp_start_x < self.ramp_end_x < self.x_bounds[1]:
            raise ValueError("The ramp must lie strictly inside the terrain X bounds")
        if self.platform_height <= 0 or not 0 < self.transition_length <= self.ramp_length / 2:
            raise ValueError("The platform height and valid rounded transition length must be positive")
        if not 0 < self.boundary_margin < min(self.size) / 2:
            raise ValueError("The task boundary margin must leave a nonempty terrain interior")
        if self.reset_x_range[0] > self.reset_x_range[1] or self.reset_y_range[0] > self.reset_y_range[1]:
            raise ValueError("Reset ranges must be ordered")
        if self.reset_x_range[1] > self.ramp_start_x - self.boundary_margin:
            raise ValueError("The reset region must lie fully on the flat approach, clear of the ramp")
        reset_corners = np.array([(x, y) for x in self.reset_x_range for y in self.reset_y_range], dtype=float)
        if not np.all(self.contains(reset_corners[:, 0], reset_corners[:, 1], margin=self.boundary_margin)):
            raise ValueError("The reset region must lie inside the safe terrain boundary")
        if self.goal_xy[0] < self.ramp_end_x + self.boundary_margin or not self.contains(
            *self.goal_xy, margin=self.boundary_margin
        ):
            raise ValueError("The goal must lie on the platform inside the safe terrain boundary")


DEFAULT_GEOMETRY = SlopePlatformGeometryCfg()


def height_at(x, y=0.0, geometry: SlopePlatformGeometryCfg = DEFAULT_GEOMETRY):
    """Analytic terrain height in metres, with scalar/NumPy broadcasting.

    The flat endpoint heights extend beyond X bounds for stable reward
    computation in a boundary-crossing step. Use ``geometry.contains`` for
    terrain membership: outside the finite XY field there is no support.
    """
    x_array, _ = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
    distance = np.clip(x_array - geometry.ramp_start_x, 0.0, geometry.ramp_length)
    blend = geometry.transition_length
    grade = geometry.max_grade
    result = np.where(
        distance < blend,
        grade * distance**2 / (2.0 * blend),
        np.where(
            distance > geometry.ramp_length - blend,
            geometry.platform_height - grade * (geometry.ramp_length - distance) ** 2 / (2.0 * blend),
            grade * (distance - blend / 2.0),
        ),
    )
    return float(result) if result.ndim == 0 else result


@configclass
class SlopePlatformTerrainGeneratorCfg(TerrainGeneratorCfg):
    """Existing procedural HField API; no new terrain plugin or assets needed."""

    height_scale: float = DEFAULT_GEOMETRY.platform_height
    geometry: SlopePlatformGeometryCfg = DEFAULT_GEOMETRY

    def validate(self) -> None:
        super().validate()
        self.geometry.validate()
        if self.height_scale < self.geometry.platform_height:
            raise ValueError("height_scale must cover the platform height in metres")

    def generate(self, size: tuple[float, float], shape: tuple[int, int]) -> np.ndarray:
        self.validate()
        if tuple(size) != self.geometry.size:
            raise ValueError("HField size must match the shared course geometry size")
        if len(shape) != 2 or any(not isinstance(n, int) or n < 2 for n in shape):
            raise ValueError("HField shape must contain two integers >= 2")
        xs = np.linspace(-size[0] / 2.0, size[0] / 2.0, shape[1])
        profile = height_at(xs, geometry=self.geometry) / self.height_scale
        return np.ascontiguousarray(np.broadcast_to(profile[None, :], shape), dtype=np.float32)


def make_terrain_asset(
    geometry: SlopePlatformGeometryCfg = DEFAULT_GEOMETRY, *, height_scale: float | None = None
) -> ProceduralHFieldAssetCfg:
    """Build the scene asset with one authoritative geometry definition.

    Bind the returned asset as ``scene.assets.terrain`` and use
    ``HFieldTerrainCfg(hfield="terrain")`` as ``scene.objs.floor``. Keep the
    floor object name so the existing four foot contact sensors keep working.
    """
    geometry.validate()
    return ProceduralHFieldAssetCfg(
        generator=SlopePlatformTerrainGeneratorCfg(
            geometry=geometry,
            height_scale=geometry.platform_height if height_scale is None else height_scale,
        ),
        size=geometry.size,
        shape=geometry.shape,
    )
