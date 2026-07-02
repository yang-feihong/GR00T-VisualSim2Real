from __future__ import annotations

import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
LOW_LEVEL_ROOT = REPO_ROOT / "gr00t/rl/third_party/visual_wholebody_lowlevel"
SCRIPTS_ROOT = LOW_LEVEL_ROOT / "legged_gym/scripts"

for path in (str(SCRIPTS_ROOT), str(LOW_LEVEL_ROOT)):
    if path in sys.path:
        sys.path.remove(path)
    sys.path.insert(0, path)

from legged_gym.scripts.manip_loco_interface import ManipLoco_Policy, build_arg_parser
from legged_gym.utils.isaaclab_app import add_app_launcher_args, launch_app


def main():
    parser = build_arg_parser()
    add_app_launcher_args(parser)
    parser.add_argument("--num_steps", type=int, default=120)
    parser.add_argument("--trace_path", type=str, default="/tmp/b2z1_third_party_lowlevel_trace.pt")
    parser.add_argument("--trace_steps", type=str, default="0,1,2,3,4,5,10,20,50,100")
    args = parser.parse_args()

    simulation_app = launch_app(args)
    runner = ManipLoco_Policy(args)
    env = runner.env

    trace_steps = {int(part) for part in args.trace_steps.split(",") if part.strip()}
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
        env.commands[:, :3] = 0.0
        with torch.no_grad():
            actions = runner.policy(runner.obs.detach(), hist_encoding=True)
        runner.obs, *_ = env.step(actions.detach())
        env.commands[:, :3] = 0.0
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
