import torch

from gr00t.rl.auto_open.tensor_fsm import TensorDoorFSM


def test_opposed_grasp_does_not_reject_axially_offset_contact_points():
    fsm = TensorDoorFSM.__new__(TensorDoorFSM)
    fsm.handle_body_idx = 0
    fsm.handle_center_local = torch.zeros(1, 3)
    fsm.handle_axis_local = torch.tensor([[1.0, 0.0, 0.0]])
    fsm.door_normal = torch.tensor([[0.0, 1.0, 0.0]])
    fsm.handle_radius = torch.tensor([0.013])
    handle_state = torch.tensor([[[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]]])
    contacts = torch.tensor(
        [[[-0.015, 0.0, -0.013], [0.015, 0.0, 0.013]]],
    )
    valid = fsm._opposed_grasp({"door_body_state_w": handle_state}, torch.tensor([0]), contacts)
    assert valid.item()
    assert (contacts[0, 1, 0] - contacts[0, 0, 0]).item() > 0.020
