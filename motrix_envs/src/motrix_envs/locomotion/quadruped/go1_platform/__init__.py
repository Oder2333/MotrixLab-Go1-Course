# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Register the task configuration before its environment implementation."""

from . import cfg, env  # Registration side effects, in dependency order.
