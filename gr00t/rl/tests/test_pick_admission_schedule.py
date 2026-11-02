"""Admission scheduling preserves row budgets and counter-keyed GPU layout seeds."""

from types import SimpleNamespace

import pytest
import torch
from gr00t.rl.pick.admission_schedule import AdmissionSchedule

from gr00t.rl.pick.basket import BasketConfig
from gr00t.rl.pick.basket_initialization import resample_basket_payloads
from gr00t.rl.pick.basket_layout import sample_layouts


def plane(device):
    boxes = torch.zeros(1, 8, 3, device=device)
    return SimpleNamespace(
        boxes=boxes,
        floor_lower=boxes.new_tensor([-0.3, -0.2, 0.0]),
        floor_upper=boxes.new_tensor([0.3, 0.2, 0.1]),
        rim_z=boxes.new_tensor(0.3),
        source="test",
        support=lambda xy: (xy.new_full(xy.shape[:-1], 0.1), xy.new_ones(xy.shape[:-1])),
    )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("seeds", [[7, 9], [2**63 - 3, 2**32 + 9]])
def test_gpu_layout_seed_override_and_mixed_attempts_match_scalar_calls(device, seeds):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("GPU layouts")
    basket = plane(device)
    source = [[{"name": "target", "size_m": [0.04, 0.04, 0.04]}]] * 2
    attempts = torch.tensor([1, 3], device=device, dtype=torch.int64)
    random_seeds = (torch.tensor(seeds, device=device) + attempts * 0x4241534B) & 0x7FFFFFFFFFFFFFFF
    config = BasketConfig(min_objects=1, max_objects=1, placement_candidates=32)
    expected = sample_layouts(
        source,
        basket,
        [(seeds[0] + 0x4241534B) % (2**63), (seeds[1] + 3 * 0x4241534B) % (2**63)],
        config,
        counts=[1, 1],
    )
    actual = sample_layouts(
        source, basket, seeds, config, counts=[1, 1], randomization_seeds=random_seeds
    )
    assert actual == expected
    episodes = [
        {
            "seed": seed,
            "asset": source[i][0],
            "distractors": [],
            "base_xy": [0, 0],
            "base_yaw": 0.0,
            "initial_arm_joints": [0.0] * 6,
            "support_z": 0.1,
            "support_z_local": 0.1,
            "object_position": [0, 0, 0.12],
            "object_yaw": 0.0,
            "basket_sampling": {"object_gap_m": 0.006, "wall_gap_m": 0.006},
        }
        for i, seed in enumerate(seeds)
    ]
    combined = resample_basket_payloads(episodes, [0, 1], basket, attempts)
    assert combined[0] == resample_basket_payloads(episodes, [0], basket, 1)[0]
    assert combined[1] == resample_basket_payloads(episodes, [1], basket, 3)[1]


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_schedule_plans_ready_rows_without_waiting_for_unready_partners(monkeypatch, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("GPU schedule")
    schedule = AdmissionSchedule(2, device, attempts=2)
    with monkeypatch.context() as scope:
        for name in ("cpu", "item", "tolist"):
            scope.setattr(
                torch.Tensor, name, lambda *args, **kwargs: pytest.fail("Scheduling host read")
            )
        schedule.advance(64)
        rows = schedule.validation_rows(torch.tensor([True, False], device=device))
        schedule.finish(rows, torch.ones_like(rows, dtype=torch.bool))
    assert rows.tolist() == [0]
    assert schedule.admitted.tolist() == [True, False]
    schedule.advance(236)
    rows = schedule.validation_rows(torch.zeros(2, device=device, dtype=torch.bool))
    assert rows.tolist() == [1]
    schedule.finish(rows, torch.zeros_like(rows, dtype=torch.bool))
    assert schedule.layout_rows().tolist() == [1]
    reset = schedule.proposals(rows, torch.ones_like(rows, dtype=torch.bool))
    assert reset.tolist() == [1] and schedule.attempt_index.tolist() == [0, 1]
    assert schedule.age.tolist() == [64, 0]
    schedule.advance(300)
    rows = schedule.validation_rows(torch.zeros(2, device=device, dtype=torch.bool))
    schedule.finish(rows, torch.zeros_like(rows, dtype=torch.bool))
    assert schedule.exhausted.tolist() == [False, True]
    assert not schedule.active.any()


def test_schedule_consumes_failed_proposals_without_moving_other_rows():
    schedule = AdmissionSchedule(2, "cpu", attempts=3)
    rows = torch.tensor([0])
    schedule.finish(rows, torch.tensor([False]))
    for expected in (1, 2):
        assert schedule.proposals(rows, torch.tensor([False])).numel() == 0
        assert schedule.attempt_index.tolist() == [expected, 0]
        assert schedule.age.tolist() == [0, 0]
    assert schedule.exhausted.tolist() == [True, False]
    assert schedule.waiting.tolist() == [False, True]


def test_schedule_deadline_caps_next_shared_step_without_delaying_it_to_a_poll():
    schedule = AdmissionSchedule(2, "cpu", cadence=128)
    schedule.advance(280)
    assert schedule.next_wait() == 20
    schedule.advance(20)
    rows = schedule.validation_rows(torch.zeros(2, dtype=torch.bool))
    assert rows.tolist() == [0, 1]


def test_schedule_never_extends_the_original_shared_physical_budget():
    schedule = AdmissionSchedule(2, "cpu", attempts=1)
    schedule.advance(300)
    assert schedule.next_wait() == 0
    schedule.expire_shared_budget()
    assert schedule.exhausted.all() and not schedule.active.any()
    with pytest.raises(ValueError, match="Shared physical"):
        schedule.advance(1)


def test_schedule_preserves_the_external_final_attempt_warmup_budget():
    schedule = AdmissionSchedule(2, "cpu", attempts=2, warmup_steps=80, final_warmup_steps=300)
    assert schedule.max_physics_steps == 380
    schedule.advance(80)
    rows = schedule.validation_rows(torch.zeros(2, dtype=torch.bool))
    schedule.finish(rows, torch.tensor([False, True]))
    schedule.proposals(torch.tensor([0]), torch.tensor([True]))
    assert schedule.limit.tolist() == [300, 80]
    assert schedule.age.tolist() == [0, 80]


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_ready_row_decisions_do_not_depend_on_partner_readiness(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("GPU schedule")
    batch = AdmissionSchedule(2, device)
    solo = AdmissionSchedule(1, device)
    for scheduler in (batch, solo):
        scheduler.advance(64)
        ready = torch.zeros_like(scheduler.waiting)
        ready[0] = True
        rows = scheduler.validation_rows(ready)
        scheduler.finish(rows, torch.zeros_like(rows, dtype=torch.bool))
        scheduler.proposals(rows, torch.ones_like(rows, dtype=torch.bool))
        scheduler.advance(32)
        ready[0] = True
        rows = scheduler.validation_rows(ready)
        scheduler.finish(rows, torch.ones_like(rows, dtype=torch.bool))
    for field in (
        "age",
        "attempt_index",
        "admitted",
        "needs_layout",
        "exhausted",
        "waiting",
        "limit",
    ):
        torch.testing.assert_close(
            getattr(batch, field)[0], getattr(solo, field)[0], rtol=0, atol=0
        )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_eight_step_checks_preserve_readiness_and_ignore_partner_count(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("GPU schedule")
    for count in (1, 4, 100):
        schedule = AdmissionSchedule(count, device, cadence=8)
        assert schedule.next_wait() == 8
        schedule.advance(35)
        ready = torch.zeros(count, device=device, dtype=torch.bool)
        ready[0] = True
        assert schedule.validation_rows(ready).numel() == 0
        assert schedule.next_wait() == 5
        schedule.advance(5)
        selected = schedule.validation_rows(ready)
        torch.testing.assert_close(selected, torch.tensor([0], device=device))
        schedule.finish(selected, torch.ones_like(selected, dtype=torch.bool))
        assert schedule.admitted[0] and schedule.age[0] == 40
        assert schedule.max_physics_steps == 16 * 300
