"""Scene-dependent GPU matching against asset-independent low-reach measurements."""

from pathlib import Path

import torch

from gr00t.rl.pick.measured_commands import read_measured_report
from gr00t.rl.pick.reach_calibration import measured_grasp_matches
from gr00t.rl.pick.station import select_initial_pick_station

COMMAND_CATALOG = Path(__file__).resolve().parents[1] / "data/pick_reach/low_commands_v2.json"


class MeasuredLowReachPlanner:
    """Offer measured low commands, retaining normal path and physical acceptance."""

    def __init__(self, episode, checkpoint_sha256, *, catalog=COMMAND_CATALOG):
        self.episode = episode
        self.catalog = Path(catalog)
        self.orientation = None
        self.depth = 0.0
        self.catalog_sha256 = None
        self.compatible = False
        if self.catalog.is_file():
            metadata, digest = read_measured_report(self.catalog)
            self.compatible = (
                metadata.get("schema") == "pick_measured_low_commands_v1"
                and metadata.get("checkpoint_sha256") == checkpoint_sha256
                and bool(checkpoint_sha256)
            )
            self.catalog_sha256 = digest

    def station(self, state, destination, size, supports, limits, collision, geometry):
        bank = None
        if self.compatible:
            bank = measured_grasp_matches(
                [self.catalog],
                None,
                None,
                device=state["ee_pos"].device,
                episode=self.episode,
                state=state,
                collision=collision,
                geometry=geometry,
            )
        if bank is None or not bool(bank["feasible"].any()):
            pose, orientation, joints, report = select_initial_pick_station(
                state,
                destination,
                size,
                supports,
                limits,
                collision,
                geometry,
            )
            report["measured_reach_fallback"] = (
                "no_scene_safe_measured_match"
                if self.compatible
                else "catalog_missing_or_policy_mismatch"
            )
            return pose, orientation, joints, report
        distance = (bank["stations"][:, :2] - state["root_states"][:, :2]).norm(dim=-1)
        selected = distance.masked_fill(~bank["feasible"], torch.inf).argmin()
        pose = bank["stations"][selected][None]
        self.orientation = bank["orientations"][selected][None]
        self.depth = float(bank["depth_preferences"][selected])
        return (
            pose,
            self.orientation,
            None,
            {
                "selection_policy": "shortest_scene_clear_measured_low_command",
                "selected_station": pose,
                "selected_orientation": self.orientation[0],
                "selected_sample_id": int(bank["sample_ids"][selected]),
                "selected_pocket_depth_preference": self.depth,
                "scene_safe_candidate_count": int(bank["feasible"].sum()),
                "measured_command_count": bank["measured_command_count"],
                "checkpoint_sha256": bank["checkpoint_sha256"],
                "command_catalog_sha256": self.catalog_sha256,
                "not_a_physical_grasp_guarantee": True,
            },
        )
