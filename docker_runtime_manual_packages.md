# Runtime packages manually installed during B2Z1 low-level debugging

Installed on 2026-07-20 for remote behavior-level comparison against `visual_wholebody_yfh`:

```dockerfile
RUN apt-get update && apt-get install -y --no-install-recommends \
    openssh-client \
    iputils-ping \
    netcat-openbsd \
 && rm -rf /var/lib/apt/lists/*
```

The B2Z1 low-level deployment path expects `rsl_rl` to come from the Python environment, not from a vendored `third_party/rsl_rl` source copy. Keep the package installed in the image/environment rather than adding a duplicate source tree to the repo.

Additional runtime notes from the 2026-07-24 B2Z1 low-level validation pass:

```dockerfile
ENV OMNI_KIT_ACCEPT_EULA=YES \
    ISAACSIM_ACCEPT_EULA=YES \
    ACCEPT_EULA=Y
```

- The local headless IsaacSim/IsaacLab runs use GPU 0 (`NVIDIA GeForce RTX 4090`, UUID prefix `bb9db648`) and the no-render Kit file at `gr00t/rl/apps/b2z1.isaaclab.python.headless.no_render.kit`.
- Remote behavior comparison used SSH into `yangfeihong@172.23.53.221` and `docker exec visual_wholebody_yfh`; keep `openssh-client` in the image if this comparison workflow should remain available.
- Latest trace/video artifacts from this pass:
  - `/tmp/b2z1_remote_zero_clean_trace.pt`
  - `/tmp/b2z1_local_zero_video_trace.pt`
  - `/tmp/b2z1_local_forward_video_trace.pt`
  - `artifacts/b2z1_videos/b2z1_0713_zero_isaaclab_current.mp4`
  - `artifacts/b2z1_videos/b2z1_0713_forward_isaaclab_current.mp4`
