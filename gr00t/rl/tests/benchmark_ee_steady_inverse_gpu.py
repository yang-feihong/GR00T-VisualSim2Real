"""Offline inverse regression on recorded stage trajectories; no simulator launch."""

import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np
import pyarrow.parquet as pq
import torch

from gr00t.rl.third_party.visual_wholebody_lowlevel.legged_gym.utils.ee_steady_inverse_gpu import (
    batched_steady_ee_inverse_no_gravity,
)


def recorded_inputs(dataset, seeds):
    parts = []
    for seed in seeds:
        shard = dataset / "shards" / f"episode-seed{seed}"
        paths = list(shard.glob("**/raw_audit/chunk-*/episode_*.parquet"))
        if len(paths) != 1:
            raise ValueError(f"expected one audit for seed {seed}, got {paths}")
        audit = pq.read_table(paths[0]).to_pydict()
        stages = np.asarray(audit["autoopen_stage"])
        for stage in ("PREGRASP", "CENTER_HANDLE", "PRESS_HANDLE", "PUSH_PULL_DOOR",
                      "RETURN_HANDLE", "OPEN_AND_CLEAR", "RETREAT_ARM", "RESTORE_GRIPPER"):
            rows = np.flatnonzero(stages == stage)
            if not len(rows):
                continue
            start = int(rows[0])
            poses = np.asarray(audit["ee_pose_world_xyzw"])[rows]
            roots = np.tile(np.r_[audit["root_position_world"][start],
                                  audit["root_quaternion_xyzw"][start]], (len(rows), 1))
            arms = np.tile(audit["arm_joint_position"][start], (len(rows), 1))
            parts.append((poses[:, :3], poses[:, [6, 3, 4, 5]], roots, arms))
    return [np.concatenate([part[i] for part in parts]) for i in range(4)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--seeds", nargs="+", type=int, default=[1700, 1702, 1703])
    parser.add_argument("--reference-root", type=Path)
    parser.add_argument("--cpu-checks", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    root = Path(__file__).resolve().parents[3]
    low = Path("gr00t/rl/third_party/visual_wholebody_lowlevel")
    urdf = ET.parse(root / low / "resources/robots/b2z1/urdf/b2z1.urdf")
    limits = []
    for joint in range(1, 7):
        limit = urdf.find(f"joint[@name='joint{joint}']/limit")
        limits.append([float(limit.get("lower")), float(limit.get("upper"))])
    limits = torch.tensor(limits, device="cuda", dtype=torch.float32)
    corpus = recorded_inputs(args.dataset, args.seeds)
    reference = None
    if args.reference_root:
        name = "gr00t.rl.third_party.visual_wholebody_lowlevel.legged_gym.utils._inverse_benchmark_reference"
        spec = importlib.util.spec_from_file_location(
            name, args.reference_root / low / "legged_gym/utils/ee_steady_inverse_gpu.py"
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        reference = module.batched_steady_ee_inverse_no_gravity
    report = {"gpu": torch.cuda.get_device_name(), "source": str(args.dataset),
              "seeds": args.seeds, "corpus_poses": len(corpus[0]), "results": []}
    for count in (1, 125, 1500, 12000):
        indices = np.arange(count) % len(corpus[0])
        position, orientation, roots, arms = [
            torch.as_tensor(value[indices], device="cuda", dtype=torch.float32)
            for value in corpus
        ]
        kwargs = dict(base_position_world=roots[:, :3],
                      base_orientation_world_xyzw=roots[:, 3:],
                      initial_arm_joint_positions=arms, joint_limits=limits,
                      position_tolerance_m=1e-4, orientation_tolerance=1e-4)
        batched_steady_ee_inverse_no_gravity(position, orientation, **kwargs)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(5):
            result = batched_steady_ee_inverse_no_gravity(position, orientation, **kwargs)
        torch.cuda.synchronize()
        row = {"points": count, "gpu_seconds": (time.perf_counter() - start) / 5}
        if reference:
            torch.cuda.synchronize()
            start = time.perf_counter()
            old = reference(position, orientation, **kwargs)
            torch.cuda.synchronize()
            row.update(
                reference_gpu_seconds=time.perf_counter() - start,
                max_position_difference_m=float(torch.linalg.vector_norm(
                    result.command_position_world - old.command_position_world, dim=-1).max()),
                max_joint_difference_rad=float((result.arm_joint_positions - old.arm_joint_positions).abs().max()),
                old_exact=int(old.exact.sum()), new_exact=int(result.exact.sum()),
                old_exact_lost=int((old.exact & ~result.exact).sum()),
                max_exact_position_difference_m=float(torch.linalg.vector_norm(
                    result.command_position_world[old.exact] - old.command_position_world[old.exact],
                    dim=-1).max()) if bool(old.exact.any()) else None,
            )
            if count == 12000 and args.cpu_checks:
                from gr00t.rl.third_party.visual_wholebody_lowlevel.legged_gym.utils.ee_steady_state import B2Z1SteadyEeForwardModel
                model = B2Z1SteadyEeForwardModel(
                    root / low / "resources/robots/b2z1/urdf/b2z1.urdf",
                    stiffness=10000.0, gravity_magnitude=0.0, emit_tolerance_warnings=False,
                )
                differences = (result.arm_joint_positions - old.arm_joint_positions).abs().amax(dim=-1)
                checks = []
                for index in torch.topk(differences, min(args.cpu_checks, count)).indices.cpu().tolist():
                    joints = dict.fromkeys(model.model.names[1:], 0.0)
                    joints.update(zip(model.arm_joint_names, arms[index].cpu().tolist(), strict=True))
                    joints["jointGripper"] = 0.0
                    root_q = roots[index, 3:].cpu().numpy()[[3, 0, 1, 2]]
                    try:
                        cpu = model.inverse(
                            position[index].cpu().numpy(), orientation[index].cpu().numpy(),
                            base_position_world=roots[index, :3].cpu().numpy(),
                            base_orientation_world_wxyz=root_q, joint_positions=joints,
                            max_function_evaluations=100,
                        )
                        desired = position[index].cpu().numpy()
                        old_error = float(np.linalg.norm(old.command_position_world[index].cpu().numpy() - desired))
                        new_error = float(np.linalg.norm(result.command_position_world[index].cpu().numpy() - desired))
                        checks.append({"index": index, "cpu_position_error_m": float(np.linalg.norm(cpu.position_error)),
                                       "old_position_error_m": old_error, "new_position_error_m": new_error,
                                       "new_cpu_position_difference_m": float(np.linalg.norm(result.command_position_world[index].cpu().numpy() - cpu.command_position_world))})
                    except RuntimeError as exc:
                        checks.append({"index": index, "cpu_error": str(exc)})
                row["cpu_checks"] = checks
        for index in sorted(set((0, count // 2, count - 1))):
            single_kwargs = {key: value[index:index+1] if torch.is_tensor(value) and key != "joint_limits" else value
                             for key, value in kwargs.items()}
            single = batched_steady_ee_inverse_no_gravity(
                position[index:index+1], orientation[index:index+1], **single_kwargs
            )
            torch.testing.assert_close(single.arm_joint_positions[0], result.arm_joint_positions[index],
                                       rtol=0.0, atol=0.0)
        report["results"].append(row)
        print(json.dumps(row), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
