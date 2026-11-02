"""Pick tasks use the door launcher's modes and process lease."""

from pathlib import Path

import pytest

from gr00t.rl.pick.entry import dispatch


@pytest.mark.parametrize("task", ["pick_external", "pick_basket"])
@pytest.mark.parametrize("mode", ["debug", "canonical", "official"])
def test_collection_does_not_dispatch_into_an_independent_pick_workflow(task, mode):
    with pytest.raises(ValueError, match="run_auto_open.sh directly"):
        dispatch([], {"AUTO_OPEN_TASK": task, "AUTO_OPEN_MODE": mode})
    launcher = (Path(__file__).resolve().parents[1] / "scripts/run_auto_open.sh").read_text()
    assert 'if [[ "${1:-}" == --pick-tool ]]' in launcher
    assert '--task_kind "${AUTO_OPEN_TASK}"' in launcher
    assert "-m gr00t.rl.batch_workflow" in launcher


def test_standalone_shell_entry_is_removed():
    assert not (Path(__file__).resolve().parents[1] / "scripts" / ("run_" + "pick.sh")).exists()


def test_fixed_entry_pins_working_directory_before_python_module_dispatch():
    launcher = (Path(__file__).resolve().parents[1] / "scripts/run_auto_open.sh").read_text()
    pinned = launcher.index('cd "${REPO_ROOT}"')
    assert pinned < launcher.index('if [[ "${1:-}" == --pick-tool ]]')
    assert pinned < launcher.index("-m gr00t.rl.batch_workflow")


def test_pick_tools_are_available_behind_same_entry():
    assert dispatch(["--pick-tool", "test", "-q"], {}) == ("pytest", ["-q"])
    assert dispatch(["--pick-tool", "report", "--help"], {}) == (
        "gr00t.rl.pick.batch_report",
        ["--help"],
    )


def test_pick_collection_defaults_to_non_mmap_usd_reader_before_rank_launch():
    launcher = (Path(__file__).resolve().parents[1] / "scripts/run_auto_open.sh").read_text()
    setting = 'export USDC_USE_PREAD="${USDC_USE_PREAD:-1}"'
    assert setting in launcher
    assert launcher.index(setting) < launcher.index("-m gr00t.rl.batch_workflow")
    assert 'if [[ "${AUTO_OPEN_TASK}" != door ]]; then\n' in launcher
    assert "USDC_USE_PREAD; do" in launcher


@pytest.mark.parametrize(
    "args,env",
    [
        ([], {"AUTO_OPEN_TASK": "invalid"}),
        ([], {"AUTO_OPEN_TASK": "pick_external", "AUTO_OPEN_MODE": "invalid"}),
        (["--mode=debug"], {"AUTO_OPEN_TASK": "pick_external"}),
        (["--pick-tool"], {}),
        (["--pick-tool", "unknown"], {}),
    ],
)
def test_invalid_or_conflicting_dispatch_is_rejected(args, env):
    with pytest.raises(ValueError):
        dispatch(args, env)
