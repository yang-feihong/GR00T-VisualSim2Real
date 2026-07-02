# B2Z1 Isaac Sim Image Dependencies

These notes record the manual package fixes used while validating
`gr00t/rl/config/robot/b2z1/README.md` stages 1 and 2.

The base environment had Isaac Sim 5.1 pip packages installed, but was missing
runtime Kit app/cache packages needed by `SimulationApp` and IsaacLab USD/URDF
conversion.

Recommended Dockerfile additions:

```bash
apt-get update && apt-get install -y --no-install-recommends \
  libxt6 \
  libglu1-mesa
```

```bash
python -m pip install \
  isaacsim-app==5.1.0.0 \
  isaacsim-extscache-kit-sdk==5.1.0.0 \
  isaacsim-asset==5.1.0.0 \
  isaacsim-robot-setup==5.1.0.0 \
  isaacsim-extscache-kit==5.1.0.0 \
  isaacsim-extscache-physics==5.1.0.0
```

Runtime environment variables used for non-interactive container runs:

```bash
export OMNI_KIT_ACCEPT_EULA=YES
export HOME=/tmp/isaac_home
export XDG_CACHE_HOME=/tmp/isaac_cache
```

Headless/no-render validation command used when the host NVIDIA driver is not
usable from the container:

```bash
HOME=/tmp/isaac_home XDG_CACHE_HOME=/tmp/isaac_cache OMNI_KIT_ACCEPT_EULA=YES \
python -u gr00t/rl/scripts/debug_b2z1_fixed_command.py \
  +exp=wbmanip/door_open_b2z1_lstm \
  num_envs=1 \
  headless=True \
  simulator.config.render_results=False \
  simulator.config.cameras.enable_cameras=False \
  'simulator.config.cameras.camera_types=[]' \
  simulator.config.sim.render_interval=1000000 \
  robot.b2z1_command.lowlevel_policy_path=null \
  '+fixed_command=[0,0,0,0,0,0,0,0,0,0]' \
  +num_debug_steps=120
```

Notes:

- `OMNI_KIT_ACCEPT_EULA=YES` avoids Isaac Sim blocking on an interactive EULA
  prompt.
- `HOME`/`XDG_CACHE_HOME` must point to writable locations if `/root` is
  read-only.
- In the live container, `isaacsim-asset`, `isaacsim-robot-setup`,
  `isaacsim-extscache-kit`, and `isaacsim-extscache-physics` were installed
  with `--no-deps` after the main Isaac Sim packages were already present.
  This avoided an unnecessary re-download of the pinned `torch==2.7.0`
  dependency in the existing environment.
- `libxt6` fixes `libXt.so.6` errors when Kit loads windowing-related shared
  libraries.
- `libglu1-mesa` fixes `libGLU.so.1` errors when Kit loads Iray/MDL-related
  shared libraries.
- Before `isaacsim-extscache-physics` was installed, IsaacLab tried to download
  many PhysX extensions at runtime into `${HOME}/.local/share/ov/data/exts/v2`;
  this made stage 2 startup very slow and vulnerable to network stalls.
- In the live container, `nvidia-smi` failed with "couldn't communicate with the
  NVIDIA driver" and PyTorch CUDA initialization returned error 304. That is a
  host driver/container GPU exposure problem, not a Python file permission
  issue. The stage-2 debug path now falls back to IsaacLab CPU/no-render mode,
  disables visual-only markers/material randomization, uses a local ground plane
  USD, and skips Kit shutdown to avoid hanging in the broken driver path after
  the debug steps complete.
- Installing `isaacsim-app` pulls Isaac Sim pinned transitive versions such as
  `click==8.1.7`, `psutil==5.9.8`, `typing_extensions==4.12.2`, and
  `starlette==0.45.3`. In this live environment that produced conflicts with
  some already-installed packages. A clean image should install Isaac Sim and
  IsaacLab dependencies in one layer/order to avoid resolver drift.
