# DATA Fisheye Border Augmentation

`AutoOpenStandardEpisodeCollector` applies `FisheyeBorderAugmenter` after sensor
readback and before dataset rotation and H.264 encoding. Official dual replay
(`base_0_rgb`, `left_wrist_0_rgb`) and six-camera debug replay (`wrist`,
`trunk_front`) use this same path. The four diagnostic cameras are excluded,
even if accidentally given a fisheye spec. Non-fisheye and disabled masks are
also excluded. Direct diagnostic/preview rendering is unchanged.

The manager supplies the effective parsed spec for the episode environment,
not whichever environment last occupied a pooled rendering slot. The existing
renderer defines alpha as `clip((radius-distance)/feather, 0, 1)` when feather
is positive, otherwise `distance <= radius`. Only zero-alpha pixels are
replaced. All scene pixels, including the complete feather band, are bitwise
unchanged before video encoding. The nominal source mask is centered at
`(320,240)`, radius `262`, feather `2`, at `640x480`; this is distinct from
the OpenCV principal point `(326.60297,249.30456)`. Neither is changed by this
augmentation. Effective episode mask values always take precedence.

The reference `00000000.jpg` has textured dark housing rings. This motivates
diversity, not reproduction of that housing. Styles are sampled independently
per episode seed and source camera name:

| Style | Probability | Components |
| --- | --- | --- |
| black | 20% | Exactly zero |
| dark_neutral | 15% | Dark level and fine noise |
| mottled | 25% | Dark level, smooth spatial mottling, noise |
| annular | 25% | Mottling, radial-distance rings, noise |
| radial_fine | 15% | Mottling, rings, angular fine striations, noise |

Uniform parameter ranges, in 8-bit RGB levels/pixels unless noted:
base luminance 5-42; ring amplitude 2-14 and period 5-28 px; mottle amplitude
2-10 and scale 18-65 px; fine radial amplitude 1-5 and integer angular
frequency 70-240; Gaussian noise standard deviation 0.5-2.5; independent RGB
gains 0.90-1.10; phases 0-2*pi. Inactive components have amplitude zero.
The outside-only transition ramp is 2-6 px. Output is rounded and clipped
to 0-80. The ramp does not shrink, expand, or blend into the scene mask.

Randomness uses local NumPy PCG64 keyed by SHA256 of version, decimal episode
seed and camera name, separated by NUL bytes. Global RNG state, batch size,
environment index, frame index, labels and scene content are not inputs.
Camera keys remain their actual source names; debug aliases are separately
keyed but follow identical distributions. Each texture is generated once and
is static throughout the episode: no temporal flicker. CPU generation and
cached masked assignment occur only at the existing CPU dataset/video boundary,
not in online simulation or the GPU control path. No per-frame random sampling
or extra GPU synchronization is added.

Actual per-camera parameters, version, RNG key, style, source shape and mask
are stored under `fisheye_border_augmentation` in episode provenance and hence
`meta/simulation_provenance.json` and per-episode reports. This is an appearance
transform, not a calibration edit. Lossy H.264 can change pixels after encoding;
the exact valid-pixel guarantee applies to RGB arrays before encoding.

## Simulator-Free Verification

```bash
export PYTHONPATH="$PWD:$PWD/gr00t/rl/third_party/visual_wholebody_lowlevel${PYTHONPATH:+:$PYTHONPATH}"
python -m pytest -q gr00t/rl/tests/test_fisheye_border.py
python -m gr00t.rl.scripts.preview_fisheye_border \
  --output /tmp/fisheye_border_preview --reference 00000000.jpg
```

The preview writes a synthetic masked inspection pattern, five full-resolution
PNG samples, a contact sheet, and actual sampled parameters. The real image is
kept separately as a visual reference, not used as an assumed calibrated mask.

Verified in the existing `doorman_train` container without launching a simulator:
62 tests passed across border augmentation, collector modes, camera mount
randomization, dataset standard, and door diagnostic camera tests. Generated
and visually inspected `outputs/fisheye_border_preview/contact_sheet.png`;
the same directory contains full-resolution samples and `parameters.json`.
