"""Pick dispatch behind the shared run_auto_open.sh launcher, never a shell entry."""

import os
import runpy
import sys

TOOLS = {
    "assets": "assets",
    "assets-library": "asset_library",
    "audit-supported-assets": "supported_asset_audit",
    "prepare": "prepare",
    "prepare-basket": "basket",
    "scene-affinity": "scene_affinity",
    "sample-distractors": "distractors",
    "probe": "probe_workspace",
    "probe-batch-scenes": "scene_adapter",
    "probe-pocket": "probe_pocket",
    "probe-contact": "contact_geometry_audit",
    "probe-grasp": "grasp_quality",
    "probe-approach": "probe_approach",
    "probe-tracking": "probe_tracking",
    "calibrate-reach": "reach_calibration",
    "inspect": "inspect_run",
    "report": "batch_report",
    "replay-plan": "replay_planning",
    "retime": "recording",
}


def dispatch(arguments, environment):
    if arguments[:1] == ["--pick-tool"]:
        if len(arguments) < 2:
            raise ValueError("--pick-tool requires a tool name")
        tool, *remaining = arguments[1:]
        if tool == "test":
            return "pytest", remaining
        if tool in ("lint", "format"):
            return "ruff", ["check" if tool == "lint" else "format", *remaining]
        if tool == "probe-policy":
            return "gr00t.rl.pick.probe_policy", remaining
        if tool not in TOOLS:
            raise ValueError(f"Unknown pick tool: {tool}")
        return f"gr00t.rl.pick.{TOOLS[tool]}", remaining
    raise ValueError(
        "Collection tasks use run_auto_open.sh directly; this module dispatches tools only"
    )


def main():
    try:
        module, arguments = dispatch(sys.argv[1:], os.environ)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    sys.argv = [module, *arguments]
    runpy.run_module(module, run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main()
