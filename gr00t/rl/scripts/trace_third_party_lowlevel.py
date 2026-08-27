from __future__ import annotations

import os
import sys
from pathlib import Path

import torch


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[3] if len(SCRIPT_PATH.parents) > 3 else Path.cwd()
LOW_LEVEL_ROOT = Path(
    os.environ.get(
        "VISUAL_WHOLEBODY_LOW_LEVEL_ROOT",
        REPO_ROOT / "gr00t/rl/third_party/visual_wholebody_lowlevel",
    )
).resolve()
SCRIPTS_ROOT = LOW_LEVEL_ROOT / "legged_gym/scripts"

for path in (str(SCRIPTS_ROOT), str(LOW_LEVEL_ROOT)):
    if path in sys.path:
        sys.path.remove(path)
    sys.path.insert(0, path)

from legged_gym.scripts.manip_loco_interface import ManipLoco_Policy, build_arg_parser
from legged_gym.utils.isaaclab_app import add_app_launcher_args, launch_app

try:
    from gr00t.rl.scripts.b2z1_physics_fixture import (
        JOINT_POS,
        POLICY_ACTION,
        ROOT_POS,
        EE_GOAL_CART,
        VELOCITY_COMMAND,
        canonical_root_state,
        joint_state_by_name,
        snapshot,
        tensor,
    )
except ModuleNotFoundError:
    from b2z1_physics_fixture import (
        JOINT_POS,
        POLICY_ACTION,
        ROOT_POS,
        EE_GOAL_CART,
        VELOCITY_COMMAND,
        canonical_root_state,
        joint_state_by_name,
        snapshot,
        tensor,
    )


class TraceManipLocoPolicy(ManipLoco_Policy):
    def configure_env_cfg(self, env_cfg):
        env_cfg.sim.device = str(self.args.device)


def run_open_loop(env, policy, trace_path: str, num_substeps: int, root_height: float):
    root_pos = (ROOT_POS[0], ROOT_POS[1], root_height)
    root_state = tensor((*root_pos, 1.0, 0.0, 0.0, 0.0, *([0.0] * 6)), env.device)
    joint_pos = joint_state_by_name(env.dof_names, JOINT_POS, env.device)
    joint_vel = torch.zeros_like(joint_pos)
    env.robot.write_root_state_to_sim(root_state)
    env.robot.write_joint_state_to_sim(joint_pos, joint_vel)
    env.robot.reset()
    env.scene.write_data_to_sim()
    env.scene.update(dt=env.cfg.sim.dt)

    # Joint state alone is insufficient: Isaac Lab retains actuator commands
    # from reset/IK. Flush a common command state without advancing physics.
    env.actions.zero_()
    env.target_pos.copy_(env.default_dof_pos)
    env._arm_pos_targets_for_step = env.default_dof_pos[:, env.arm_joint_ids].clone()
    env.gripper_pos_targets.copy_(joint_pos[:, env.gripper_joint_ids])
    env._apply_action()
    env.scene.write_data_to_sim()

    data = env.robot.data
    metadata = {
        "body_names": list(env.body_names),
        "articulation_dof_names": list(data.joint_names),
        "initial_root_state": snapshot(canonical_root_state(env.root_states)),
        "initial_joint_pos": snapshot(env.dof_pos),
        "initial_joint_vel": snapshot(env.dof_vel),
        "initial_body_pos": snapshot(data.body_pos_w),
        "initial_body_quat": snapshot(data.body_quat_w),
        "initial_body_lin_vel": snapshot(data.body_lin_vel_w),
        "initial_body_ang_vel": snapshot(data.body_ang_vel_w),
        "body_mass": snapshot(data.default_mass),
        "body_inertia": snapshot(data.default_inertia),
        "joint_armature": snapshot(data.default_joint_armature),
        "joint_friction": snapshot(data.default_joint_friction_coeff),
        "default_joint_pos": snapshot(env.default_dof_pos),
        "joint_pos_limits": snapshot(data.joint_pos_limits),
        "joint_vel_limits": snapshot(data.joint_vel_limits),
        "joint_effort_limits": snapshot(data.joint_effort_limits),
    }

    env._teleop_ee_synced_once = True
    env.teleop_mode = True
    env.teleop_raw_ee_goal_cart.copy_(tensor(EE_GOAL_CART, env.device))
    env.teleop_raw_ee_goal_orn_delta_rpy.zero_()
    env._teleop_inputs_dirty = True
    env._refresh_ee_and_jacobian_for_ik()
    env._update_curr_ee_goal(refresh_base_yaw=False)
    env.teleop_mode = False
    ik_alignment = {
        "goal_position": snapshot(env.curr_ee_goal_cart_world),
        "goal_orientation_wxyz": snapshot(env.ee_goal_orn_quat),
        "arm_target": snapshot(env._get_arm_pos_targets()),
    }

    env.commands[:, :3] = tensor(VELOCITY_COMMAND, env.device)
    env.curr_ee_goal_cart.copy_(tensor(EE_GOAL_CART, env.device))
    env.action_history_buf.zero_()
    env.obs_history_buf.zero_()
    env.episode_length_buf.zero_()
    env._reset_gait_state(torch.arange(env.num_envs, device=env.device))
    policy_obs = env._get_observations()["policy"].clone()
    with torch.no_grad():
        policy_output = policy(policy_obs, hist_encoding=True)

    actions = tensor(POLICY_ACTION, env.device)
    records = []
    for _ in range(num_substeps):
        env.actions.copy_(actions)
        env.target_pos.copy_(env.default_dof_pos)
        env._arm_pos_targets_for_step.copy_(env.default_dof_pos[:, env.arm_joint_ids])
        env.gripper_pos_targets.copy_(joint_pos[:, env.gripper_joint_ids])
        env._apply_action()
        requested = env.torques.clone()
        position_target = env.robot.data.joint_pos_target.clone()
        control_target = env.target_pos.clone()
        env.scene.write_data_to_sim()
        env.sim.step(render=False)
        env.scene.update(dt=env.cfg.sim.dt)
        records.append(
            {
                "root_state": snapshot(canonical_root_state(env.root_states)),
                "joint_pos": snapshot(env.dof_pos),
                "joint_vel": snapshot(env.dof_vel),
                "requested_torque": snapshot(requested),
                "position_target": snapshot(position_target),
                "control_target": snapshot(control_target),
                "applied_torque": snapshot(env.robot.data.applied_torque),
            }
        )
    torch.save(
        {
            "dof_names": list(env.dof_names),
            "metadata": metadata,
            "policy_alignment": {
                "proprio": snapshot(policy_obs[:, : int(env.cfg.env.num_proprio)]),
                "observation": snapshot(policy_obs),
                "action": snapshot(policy_output),
            },
            "ik_alignment": ik_alignment,
            "records": records,
        },
        trace_path,
    )
    print(f"wrote open-loop trace {trace_path}")


def main():
    parser = build_arg_parser()
    add_app_launcher_args(parser)
    parser.add_argument("--num_steps", type=int, default=120)
    parser.add_argument("--velocity_command", type=float, nargs=3, default=[0.0, 0.0, 0.0])
    parser.add_argument("--trace_path", type=str, default="/tmp/b2z1_third_party_lowlevel_trace.pt")
    parser.add_argument("--trace_steps", type=str, default="0,1,2,3,4,5,10,20,50,100")
    parser.add_argument("--zero_initial_command_obs", action="store_true")
    parser.add_argument("--open_loop_trace_path", type=str, default=None)
    parser.add_argument("--open_loop_substeps", type=int, default=4)
    parser.add_argument("--open_loop_root_height", type=float, default=ROOT_POS[2])
    args = parser.parse_args()

    simulation_app = launch_app(args)
    runner = TraceManipLocoPolicy(args)
    env = runner.env

    if args.open_loop_trace_path:
        run_open_loop(
            env,
            runner.policy,
            args.open_loop_trace_path,
            args.open_loop_substeps,
            args.open_loop_root_height,
        )
        simulation_app.close()
        return

    if args.zero_initial_command_obs:
        num_prop = int(env.cfg.env.num_proprio)
        num_priv = int(env.cfg.env.num_priv)
        command_slice = slice(57, 60)
        runner.obs[:, command_slice] = 0.0
        history = runner.obs[:, num_prop + num_priv :].view(
            env.num_envs, int(env.cfg.env.history_len), num_prop
        )
        history[:, :, command_slice] = 0.0

    trace_steps = {int(part) for part in args.trace_steps.split(",") if part.strip()}
    velocity_command = torch.tensor(args.velocity_command, device=env.device, dtype=torch.float32)
    trace = {
        "meta": {
            "dof_names": list(env.dof_names),
            "body_names": list(env.body_names),
            "policy_joint_ids": env.policy_joint_ids.detach().cpu(),
            "default_dof_pos": env.default_dof_pos[0].detach().cpu(),
            "dt": env.dt,
        },
        "steps": {},
    }

    for step in range(int(args.num_steps)):
        env.commands[:, :3] = velocity_command
        with torch.no_grad():
            actions = runner.policy(runner.obs.detach(), hist_encoding=True)
        runner.obs, *_ = env.step(actions.detach())
        env.commands[:, :3] = velocity_command
        if step in trace_steps:
            trace["steps"][step] = {
                "root_state": env.root_states[0].detach().cpu(),
                "dof_pos": env.dof_pos[0].detach().cpu(),
                "dof_vel": env.dof_vel[0].detach().cpu(),
                "base_lin_vel": env.base_lin_vel[0].detach().cpu(),
                "base_ang_vel": env.base_ang_vel[0].detach().cpu(),
                "obs": runner.obs[0].detach().cpu(),
                "actions": actions[0].detach().cpu(),
                "effective_actions": env.actions[0].detach().cpu(),
                "torques": env.torques[0].detach().cpu(),
                "commands": env.commands[0, :3].detach().cpu(),
            }
        if step % 50 == 0:
            print(
                f"step={step} root={env.root_states[0, :3].detach().cpu().tolist()} "
                f"cmd={env.commands[0, :3].detach().cpu().tolist()}"
            )

    trace_path = Path(args.trace_path)
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(trace, trace_path)
    print(f"wrote {trace_path}")
    print(
        "summary initial={} final={} delta={}".format(
            trace["steps"][min(trace["steps"])]['root_state'][:3].tolist(),
            env.root_states[0, :3].detach().cpu().tolist(),
            (
                env.root_states[0, :3].detach().cpu()
                - trace["steps"][min(trace["steps"])]['root_state'][:3]
            ).tolist(),
        )
    )
    if simulation_app is not None:
        simulation_app.close()


if __name__ == "__main__":
    main()
