"""Task-independent device stability and named stage acceptance diagnostics."""

import math
from dataclasses import dataclass

import torch

from gr00t.rl.auto_open.motion_math import _quat_rotate
from gr00t.rl.third_party.visual_wholebody_lowlevel.legged_gym.utils.batch_invariant_math import (
    fixed_dot,
    fixed_mean,
    fixed_norm,
    fixed_normalize,
)


@dataclass(frozen=True)
class RobotStabilityProfile:
    required_frames: int = 25
    max_linear_speed_m_s: float = 0.05
    max_angular_speed_rad_s: float = 0.05


def consecutive_steps(condition, previous):
    """Uniform device-only task/stability window with reset on false."""
    return torch.where(condition, previous + 1, 0)


def update_robot_stability(
    base_states: torch.Tensor,
    settle_steps: torch.Tensor,
    *,
    required_frames=None,
    max_linear_speed_m_s=None,
    max_angular_speed_rad_s=None,
    position_error_m=None,
    max_position_error_m=None,
    yaw_error_rad=None,
    max_yaw_error_rad=None,
    min_base_height_m=None,
    min_upright_cos=None,
    door_speed_rad_s=None,
    max_door_speed_rad_s=None,
    additional_condition=None,
    profile=RobotStabilityProfile(),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Evaluate and accumulate one parameterized robot-stability window."""
    device = base_states.device
    dtype = base_states.dtype
    required_frames = profile.required_frames if required_frames is None else required_frames
    max_linear_speed_m_s = (
        profile.max_linear_speed_m_s if max_linear_speed_m_s is None else max_linear_speed_m_s
    )
    max_angular_speed_rad_s = (
        profile.max_angular_speed_rad_s
        if max_angular_speed_rad_s is None
        else max_angular_speed_rad_s
    )

    def threshold(value):
        return torch.as_tensor(value, device=device, dtype=dtype)

    stable = (fixed_norm(base_states[:, 7:10]) < threshold(max_linear_speed_m_s)) & (
        fixed_norm(base_states[:, 10:13]) < threshold(max_angular_speed_rad_s)
    )
    if position_error_m is not None:
        stable &= position_error_m < threshold(max_position_error_m)
    if yaw_error_rad is not None:
        stable &= yaw_error_rad.abs() < threshold(max_yaw_error_rad)
    if min_base_height_m is not None:
        stable &= base_states[:, 2] >= threshold(min_base_height_m)
    if min_upright_cos is not None:
        up = _quat_rotate(
            base_states[:, 3:7],
            torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype).expand(len(base_states), -1),
        )[:, 2]
        stable &= up >= threshold(min_upright_cos)
    if door_speed_rad_s is not None:
        stable &= door_speed_rad_s.abs() < threshold(max_door_speed_rad_s)
    if additional_condition is not None:
        stable &= torch.as_tensor(additional_condition, device=device, dtype=torch.bool)

    required = torch.as_tensor(required_frames, device=device, dtype=settle_steps.dtype)
    next_steps = torch.where(
        stable,
        torch.minimum(settle_steps + 1, torch.ones_like(settle_steps) * required),
        torch.zeros_like(settle_steps),
    )
    return next_steps >= required, next_steps, stable


class TensorStabilityWindow:
    """Device-only equivalent of the legacy 15-frame EE stability deque."""

    def __init__(
        self,
        num_envs: int,
        device,
        window_size: int = 15,
        max_position_span_m: float = 0.001,
        max_orientation_span_rad: float = 0.003,
    ):
        self.window_size = int(window_size)
        self.max_position_span_m = max_position_span_m
        self.max_orientation_span_rad = max_orientation_span_rad
        self.positions = torch.zeros((num_envs, window_size, 3), dtype=torch.float32, device=device)
        self.orientations = torch.zeros(
            (num_envs, window_size, 4), dtype=torch.float32, device=device
        )
        self.count = torch.zeros(num_envs, dtype=torch.int32, device=device)

    def clear(self, rows: torch.Tensor) -> None:
        self.count[rows] = 0

    def update(
        self, state: dict[str, torch.Tensor], rows: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self.positions[rows] = torch.roll(self.positions[rows], -1, dims=1)
        self.orientations[rows] = torch.roll(self.orientations[rows], -1, dims=1)
        self.positions[rows, -1] = state["ee_pos"][rows]
        self.orientations[rows, -1] = state["ee_orn"][rows]
        self.count[rows] = torch.clamp(self.count[rows] + 1, max=self.window_size)

        positions = self.positions[rows]
        position_span = fixed_norm(positions - fixed_mean(positions, dim=1, keepdim=True)).amax(
            dim=1
        )
        orientations = self.orientations[rows]
        orientations = fixed_normalize(orientations, eps=1.0e-8)
        current = orientations[:, -1]
        dots = fixed_dot(orientations, current[:, None, :]).abs()
        orientation_span = (2.0 * torch.acos(dots.clamp(0.0, 1.0))).amax(dim=1)
        stable = (
            (self.count[rows] >= self.window_size)
            & (position_span <= self.max_position_span_m)
            & (orientation_span <= self.max_orientation_span_rad)
        )
        return stable, position_span, orientation_span


class StageAcceptanceMixin:
    def record_acceptance(
        self,
        stage,
        rows: torch.Tensor,
        conditions: dict[str, tuple],
        *,
        required_streak: int = 1,
    ) -> torch.Tensor:
        """Observe named acceptance predicates without affecting behavior."""
        combined = torch.ones(len(rows), dtype=torch.bool, device=self.device)
        normalized = {}
        for name, item in conditions.items():
            passed, value, comparison, threshold = item
            passed = torch.as_tensor(passed, dtype=torch.bool, device=self.device)
            value = torch.as_tensor(value, dtype=torch.float32, device=self.device)
            if value.ndim == 0:
                value = value.expand(len(rows))
            threshold = torch.as_tensor(threshold, dtype=torch.float32, device=self.device)
            if threshold.ndim == 0:
                threshold = threshold.expand(len(rows))
            normalized[name] = (passed, value, comparison, threshold)
            combined &= passed
        # Some focused control tests intentionally construct a minimal FSM via
        # __new__ without diagnostics.  Observation must remain passive there.
        if not hasattr(self, "_acceptance"):
            return combined
        normalized["all_conditions"] = (
            combined,
            combined.to(torch.float32),
            "consecutive_true_frames>=",
            torch.full(
                (len(rows),),
                float(required_streak),
                dtype=torch.float32,
                device=self.device,
            ),
        )
        for name, (passed, value, comparison, threshold) in normalized.items():
            key = (int(stage), name)
            entry = self._acceptance.get(key)
            if entry is None:
                entry = {
                    "comparison": comparison,
                    "updated": torch.zeros(self.num_envs, dtype=torch.bool, device=self.device),
                    "current": torch.zeros(self.num_envs, dtype=torch.bool, device=self.device),
                    "true_frames": torch.zeros_like(self.stage_steps),
                    "streak": torch.zeros_like(self.stage_steps),
                    "max_streak": torch.zeros_like(self.stage_steps),
                    "sole_blocker_frames": torch.zeros_like(self.stage_steps),
                    "value": torch.full((self.num_envs,), torch.nan, device=self.device),
                    "min_value": torch.full((self.num_envs,), torch.inf, device=self.device),
                    "max_value": torch.full((self.num_envs,), -torch.inf, device=self.device),
                    "min_failed_value": torch.full((self.num_envs,), torch.inf, device=self.device),
                    "max_failed_value": torch.full(
                        (self.num_envs,), -torch.inf, device=self.device
                    ),
                    "min_value_step": torch.full_like(self.stage_steps, -1),
                    "max_value_step": torch.full_like(self.stage_steps, -1),
                    "min_failed_value_step": torch.full_like(self.stage_steps, -1),
                    "max_failed_value_step": torch.full_like(self.stage_steps, -1),
                    "failed_streak": torch.zeros_like(self.stage_steps),
                    "max_failed_streak": torch.zeros_like(self.stage_steps),
                    "max_failed_streak_end_step": torch.full_like(self.stage_steps, -1),
                    "threshold": torch.full((self.num_envs,), torch.nan, device=self.device),
                    "observed_stage_step": torch.full_like(self.stage_steps, -1),
                    "observed_stage_visit": torch.full_like(self.stage_steps, -1),
                }
                self._acceptance[key] = entry
            entry["updated"][rows] = True
            entry["observed_stage_step"][rows] = self.stage_steps[rows]
            entry["observed_stage_visit"][rows] = self.stage_visit[rows, int(stage)]
            entry["current"][rows] = passed
            entry["true_frames"][rows] += passed.to(torch.int32)
            streak = torch.where(passed, entry["streak"][rows] + 1, 0)
            entry["streak"][rows] = streak
            entry["max_streak"][rows] = torch.maximum(entry["max_streak"][rows], streak)
            entry["value"][rows] = value
            new_min = value < entry["min_value"][rows]
            new_max = value > entry["max_value"][rows]
            entry["min_value"][rows] = torch.minimum(entry["min_value"][rows], value)
            entry["max_value"][rows] = torch.maximum(entry["max_value"][rows], value)
            entry["min_value_step"][rows] = torch.where(
                new_min, self.stage_steps[rows], entry["min_value_step"][rows]
            )
            entry["max_value_step"][rows] = torch.where(
                new_max, self.stage_steps[rows], entry["max_value_step"][rows]
            )
            failed = ~passed
            new_failed_min = failed & (value < entry["min_failed_value"][rows])
            new_failed_max = failed & (value > entry["max_failed_value"][rows])
            entry["min_failed_value"][rows] = torch.where(
                failed,
                torch.minimum(entry["min_failed_value"][rows], value),
                entry["min_failed_value"][rows],
            )
            entry["max_failed_value"][rows] = torch.where(
                failed,
                torch.maximum(entry["max_failed_value"][rows], value),
                entry["max_failed_value"][rows],
            )
            entry["min_failed_value_step"][rows] = torch.where(
                new_failed_min,
                self.stage_steps[rows],
                entry["min_failed_value_step"][rows],
            )
            entry["max_failed_value_step"][rows] = torch.where(
                new_failed_max,
                self.stage_steps[rows],
                entry["max_failed_value_step"][rows],
            )
            failed_streak = torch.where(failed, entry["failed_streak"][rows] + 1, 0)
            longer_failed_streak = failed_streak > entry["max_failed_streak"][rows]
            entry["failed_streak"][rows] = failed_streak
            entry["max_failed_streak"][rows] = torch.maximum(
                entry["max_failed_streak"][rows], failed_streak
            )
            entry["max_failed_streak_end_step"][rows] = torch.where(
                longer_failed_streak,
                self.stage_steps[rows],
                entry["max_failed_streak_end_step"][rows],
            )
            entry["threshold"][rows] = threshold
            if name != "all_conditions":
                other_conditions = [
                    item[0]
                    for other_name, item in normalized.items()
                    if other_name not in (name, "all_conditions")
                ]
                if other_conditions:
                    others_passed = torch.stack(other_conditions).all(dim=0)
                    entry["sole_blocker_frames"][rows] += ((~passed) & others_passed).to(
                        torch.int32
                    )
        return combined

    def begin_acceptance_frame(self) -> None:
        """Mark the start of one observable FSM frame."""
        for entry in self._acceptance.values():
            entry["updated"].zero_()

    def acceptance_frame_rows(self, action_frame_index: int) -> list[list[dict]]:
        """Copy this frame's named acceptance inputs into long-form rows."""
        state_frame_index = int(action_frame_index) + 1
        per_env = [[] for _ in range(self.num_envs)]
        for (stage_id, name), entry in self._acceptance.items():
            env_ids = torch.where(entry["updated"])[0]
            if not len(env_ids):
                continue
            ids = env_ids.detach().cpu().tolist()
            values = entry["value"][env_ids].detach().cpu().tolist()
            thresholds = entry["threshold"][env_ids].detach().cpu().tolist()
            passed = entry["current"][env_ids].detach().cpu().tolist()
            streaks = entry["streak"][env_ids].detach().cpu().tolist()
            failed_streaks = entry["failed_streak"][env_ids].detach().cpu().tolist()
            stage_steps = entry["observed_stage_step"][env_ids].detach().cpu().tolist()
            stage_visits = entry["observed_stage_visit"][env_ids].detach().cpu().tolist()
            for offset, env_id in enumerate(ids):
                per_env[env_id].append(
                    {
                        # Acceptance is evaluated from state t[k + 1] after
                        # action k has been executed.  Preserve both clocks so
                        # the trace joins to raw audit rows without ambiguity.
                        "frame_index": state_frame_index,
                        "state_frame_index": state_frame_index,
                        "action_frame_index": int(action_frame_index),
                        "stage": self.Stage(stage_id).name,
                        "stage_id": int(stage_id),
                        "stage_step": int(stage_steps[offset]),
                        "stage_visit": int(stage_visits[offset]),
                        "condition": name,
                        "value": float(values[offset]),
                        "comparison": entry["comparison"],
                        "threshold": float(thresholds[offset]),
                        "passed": bool(passed[offset]),
                        "current_true_streak": int(streaks[offset]),
                        "current_false_streak": int(failed_streaks[offset]),
                    }
                )
        return per_env

    def acceptance_diagnostics(self, env_id: int, stage) -> dict:
        """Return the uniform per-condition report for one stage visit."""
        result = {}
        for (stage_id, name), entry in self._acceptance.items():
            if stage_id != int(stage):
                continue
            observed = int(entry["observed_stage_step"][env_id].item()) >= 0 and int(
                entry["observed_stage_visit"][env_id].item()
            ) == int(self.stage_visit[env_id, stage_id].item())
            if not observed:
                result[name] = {
                    "observed": False,
                    "current": False,
                    "value": None,
                    "threshold": None,
                    "comparison": entry["comparison"],
                    "reason": "no_acceptance_sample_in_current_stage_visit",
                    "true_frames": 0,
                    "max_streak": 0,
                    "window_satisfied": False,
                }
                continue
            min_failed = float(entry["min_failed_value"][env_id].item())
            max_failed = float(entry["max_failed_value"][env_id].item())
            result[name] = {
                "observed": True,
                "current": bool(entry["current"][env_id].item()),
                "value": float(entry["value"][env_id].item()),
                "comparison": entry["comparison"],
                "threshold": float(entry["threshold"][env_id].item()),
                "min_value": float(entry["min_value"][env_id].item()),
                "max_value": float(entry["max_value"][env_id].item()),
                "min_failed_value": min_failed if math.isfinite(min_failed) else None,
                "max_failed_value": max_failed if math.isfinite(max_failed) else None,
                "min_value_step": int(entry["min_value_step"][env_id].item()),
                "max_value_step": int(entry["max_value_step"][env_id].item()),
                "min_failed_value_step": int(entry["min_failed_value_step"][env_id].item()),
                "max_failed_value_step": int(entry["max_failed_value_step"][env_id].item()),
                "max_failed_streak": int(entry["max_failed_streak"][env_id].item()),
                "max_failed_streak_end_step": int(
                    entry["max_failed_streak_end_step"][env_id].item()
                ),
                "true_frames": int(entry["true_frames"][env_id].item()),
                "current_streak": int(entry["streak"][env_id].item()),
                "max_streak": int(entry["max_streak"][env_id].item()),
                "sole_blocker_frames": int(entry["sole_blocker_frames"][env_id].item()),
            }
            if name == "all_conditions":
                required = int(entry["threshold"][env_id].item())
                result[name]["window_satisfied"] = result[name]["max_streak"] >= required
        return result
