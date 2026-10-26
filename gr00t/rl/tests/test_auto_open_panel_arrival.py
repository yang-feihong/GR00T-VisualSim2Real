import math
from types import SimpleNamespace

import torch
import pytest

from gr00t.rl.auto_open.panel_arrival import panel_arrival_path
from gr00t.rl.auto_open.tensor_stage_planning import _quat_rotate
from gr00t.rl.auto_open.first_pull_wall import retime_first_pull_paths
from gr00t.rl.auto_open.tensor_fsm import PaddedPathBank,TensorDoorFSM


def test_yaw_first_joint_route_is_continuous_bounded_and_batch_independent():
    from gr00t.rl.auto_open.panel_arrival import panel_joint_route
    start = torch.tensor([[0.,.4,-.2,0.,0.,0.]],dtype=torch.float64)
    end = torch.tensor([[.6,.8,-.8,.3,.1,-.1]],dtype=torch.float64)
    path,count = panel_joint_route(start,end,384,yaw_first=True)
    first_steps = 30
    torch.testing.assert_close(path[:,first_steps,0],end[:,0])
    torch.testing.assert_close(path[:,first_steps,1:],start[:,1:])
    torch.testing.assert_close(path[:,count.item()-1],end)
    assert (path[:,1:]-path[:,:-1]).abs().max()<=.02+1.e-12
    batch,batch_count = panel_joint_route(start.expand(3,-1),end.expand(3,-1),384,yaw_first=True)
    torch.testing.assert_close(batch[0],path[0])
    assert int(batch_count[0])==int(count[0])


def test_height_preference_keeps_world_goal_fixed_and_does_not_mutate_state(monkeypatch):
    from gr00t.rl.auto_open.panel_arrival import panel_standing_height_preference
    captured = []
    def inverse(position,orientation,**kwargs):
        captured.append((position.clone(),kwargs['base_position_world'].clone()))
        n = len(position)
        error = torch.zeros(n,3,dtype=torch.float64)
        error[:,0] = (kwargs['base_position_world'][:,2]>.59).double()*.01
        return SimpleNamespace(arm_joint_positions=torch.zeros(n,6,dtype=torch.float64),
            position_error=error,orientation_error=torch.zeros_like(error))
    monkeypatch.setattr('legged_gym.utils.ee_steady_inverse_gpu.batched_steady_ee_inverse_no_gravity',inverse)
    roots = torch.tensor([[0.,0.,.51,1.,0.,0.,0.]],dtype=torch.float64)
    state = dict(root_states=roots.clone())
    bank = dict(targets=torch.tensor([[[.5,0.,1.2]]],dtype=torch.float64),
        quats=torch.tensor([[[1.,0.,0.,0.]]],dtype=torch.float64),entry_joints=torch.zeros(1,1,6,dtype=torch.float64))
    coordinator = SimpleNamespace(joint_limits=torch.tensor([[-3.,3.]]*6))
    fsm = SimpleNamespace(env=SimpleNamespace())
    assert not panel_standing_height_preference(coordinator,fsm,state,torch.tensor([0]),bank).any()
    torch.testing.assert_close(captured[0][0],bank['targets'][0].expand(2,-1))
    torch.testing.assert_close(captured[0][1][:,2],torch.tensor([.555,.6],dtype=torch.float64))
    torch.testing.assert_close(state['root_states'],roots)


def test_staging_path_keeps_tool_clear_and_respects_pose_rates():
    start = torch.tensor([[.5,.3,1.]],dtype=torch.float64)
    end = torch.tensor([[.13,.4,1.1]],dtype=torch.float64)
    q0,q1 = torch.tensor([[1.,0.,0.,0.]],dtype=torch.float64),torch.tensor([[0.,0.,0.,1.]],dtype=torch.float64)
    normal = torch.tensor([[1.,0.,0.]],dtype=torch.float64)
    jaw = torch.tensor([[[.1,0.,0.],[-.05,0.,0.]]],dtype=torch.float64)
    p,q,length = panel_arrival_path(start,q0,end,q1,normal,jaw)
    n = int(length[0])
    torch.testing.assert_close(p[0,0],start[0])
    torch.testing.assert_close(p[0,n-1],end[0])
    assert torch.linalg.vector_norm(p[0,1:n]-p[0,:n-1],dim=-1).max() <= .00300001
    angle = 2*torch.acos((q[0,1:n]*q[0,:n-1]).sum(-1).abs().clamp(0.,1.))
    assert angle.max() <= .02500001
    points = p[:,:,None]+_quat_rotate(q[:,:,None],jaw[:,None])
    assert points[...,0].min() >= .03-1.e-9
    shifted = start.repeat(2,1)
    shifted[1] += torch.tensor([32.,64.,0.])
    ends = end.repeat(2,1)
    ends[1] += torch.tensor([32.,64.,0.])
    pb,qb,lb = panel_arrival_path(shifted,q0.repeat(2,1),ends,q1.repeat(2,1),normal.repeat(2,1),jaw.repeat(2,1,1))
    torch.testing.assert_close(pb[0],p[0])
    torch.testing.assert_close(qb[0],q[0])
    assert lb[0] == length[0]


def test_arrival_retiming_bounds_cartesian_and_joint_steps():
    p = torch.zeros(1,32,3,dtype=torch.float64)
    p[:,1:,0] = .03
    q = torch.zeros(1,32,4,dtype=torch.float64)
    q[:,:,0] = 1.
    q[:,1:,0],q[:,1:,3] = math.cos(.1),math.sin(.1)
    joints = torch.zeros(1,32,6,dtype=torch.float64)
    joints[:,1:,0] = .04
    out,orientation,length,overflow = retime_first_pull_paths(p,q,joints,torch.tensor([2]),
        max_joint_step=.02,max_position_step=.003,max_rotation_step=.025)
    assert not overflow.any()
    assert length.item() == 11
    assert torch.linalg.vector_norm(out[0,1:11]-out[0,:10],dim=-1).max() <= .00300001


def test_contact_execution_uses_exact_cached_samples_then_holds_endpoint():
    fsm = TensorDoorFSM.__new__(TensorDoorFSM)
    fsm.device = 'cpu'
    fsm.Stage = SimpleNamespace(CONTACT_PANEL=1)
    fsm.mask = lambda stage:torch.tensor([True])
    fsm.panel_contact_plan_valid = torch.tensor([True])
    fsm.pull_door = False
    fsm.panel_arrival_path = PaddedPathBank(1,4,'cpu')
    p = torch.tensor([[[.1,.2,.3],[.101,.202,.303]]])
    q = torch.tensor([[[1.,0.,0.,0.],[1.,0.,0.,0.]]])
    fsm.panel_arrival_path.install_tensor(torch.tensor([0]),p,q,torch.tensor([2]))
    fsm.ee_command_world = torch.zeros(1,3)
    fsm.ee_orientation_command = torch.zeros(1,4)
    fsm.q_closed = torch.zeros(1)
    fsm._base_hold_command = lambda state,rows:(torch.zeros(1,3),None,None)
    received = []
    fsm._set_cartesian_targets = lambda rows,position,orientation,buffers:received.append(position.clone())
    buffers = SimpleNamespace(base=torch.zeros(1,3),gripper=torch.zeros(1,1))
    for _ in range(3):
        fsm.command_contact_panel({},buffers)
    torch.testing.assert_close(received[0],p[:,0])
    torch.testing.assert_close(received[1],p[:,1])
    torch.testing.assert_close(received[2],p[:,1])


@pytest.mark.parametrize('preferred',[True,False])
def test_arrival_accepts_production_float32_inverse_into_double_planning_buffers(monkeypatch,preferred):
    from gr00t.rl.auto_open.panel_arrival import validate_panel_arrivals
    monkeypatch.setattr('gr00t.rl.auto_open.panel_arrival.panel_standing_height_preference',
        lambda coordinator,fsm,state,rows,bank:torch.full_like(bank['targets'][...,0],preferred,dtype=torch.bool))

    coordinator = SimpleNamespace(arm_joint_ids=torch.arange(6),
        joint_limits=torch.tensor([[-3.,3.]]*6),path_diagnostics=[],
        _panel_arrival_envelope=SimpleNamespace(clearance=lambda roots,*args,**kwargs:
            torch.ones(len(roots),dtype=torch.float64)))
    def inverse(fsm,state,rows,p,q,lengths,**kwargs):
        assert state['root_states'].dtype == torch.float32
        owner,sample = torch.where(torch.arange(p.shape[1])[None]<lengths[:,None])
        n = len(owner)
        return SimpleNamespace(position_error=torch.zeros(n,3),orientation_error=torch.zeros(n,3),
            arm_joint_positions=torch.zeros(n,6),command_position_world=p[owner,sample].float(),
            command_orientation_world_wxyz=q[owner,sample].float()),owner,sample
    coordinator._inverse_pose_bank = inverse
    fsm = SimpleNamespace(device='cpu',hinge_world=torch.zeros(1,3),panel_arrival_path=PaddedPathBank(1,384,'cpu'))
    state = dict(root_states=torch.tensor([[0.,0.,.5,1.,0.,0.,0.]]),dof_pos=torch.zeros(1,6),
        ee_pos=torch.tensor([[.3,0.,1.]]),ee_orn=torch.tensor([[1.,0.,0.,0.]]),
        gripper_position=torch.zeros(1,1))
    bank = dict(targets=torch.tensor([[[.5,0.,1.]]],dtype=torch.float64),
        quats=torch.tensor([[[1.,0.,0.,0.]]],dtype=torch.float64),
        normal=torch.tensor([[1.,0.,0.]],dtype=torch.float64),
        tangent=torch.tensor([[0.,1.,0.]],dtype=torch.float64),
        jaw_local=torch.tensor([[[.1,0.,0.]]],dtype=torch.float64),
        entry_joints=torch.zeros(1,1,6,dtype=torch.float64),
        tip_surface=torch.tensor([[[.4,0.,1.]]],dtype=torch.float64))
    result = validate_panel_arrivals(coordinator,fsm,state,torch.tensor([0]),bank,
        torch.ones(1,1,92,dtype=torch.bool),torch.tensor([[2]]),
        torch.deg2rad(torch.arange(91.)+8.)[None,None],torch.ones(1,1,dtype=torch.bool))
    assert result[1].item()
    assert fsm.panel_arrival_path.positions.dtype == torch.float32
    assert fsm.panel_arrival_path.length.item()>0
