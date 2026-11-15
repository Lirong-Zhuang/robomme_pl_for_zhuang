# OpenCV + F2G grounding

This package is the only owner of object perception for the new harness path.
The in-task VLM supplies image-space `point_xy` and, only for catalog actions
that require it, `color_hint`. OpenCV turns that hint into a binary mask. F2G
turns aligned RGB, raw depth, camera intrinsics, and that mask into
`T_cam_obj` and metric `size_m`. The transform module converts the prediction
to a world-frame `ObjectEstimate`, which the action registry binds to the
ported Oracle Plus function.

## Process boundary

Keep the two Python environments separate:

- `robomme`: environment, VLM harness, OpenCV, mask generation, action calls;
- `f2g`: F2G-Pose, its CUDA extensions, RADIO, and the persistent model server.

Start the server from the RoboMME repository root with the `f2g` environment:

```bash
micromamba activate f2g
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export F2G_ROOT="$HOME/F2G-Pose"
export F2G_RADIO_REPOSITORY="$F2G_ROOT/third_party_local/RADIO"
export F2G_RADIO_CHECKPOINT="$F2G_ROOT/weights/radio-v2.5-l_half.pth.tar"
python scripts/f2g_pose_server.py \
  --checkpoint "$F2G_ROOT/weights/f2g-pose.pt"
```

The server listens only on `127.0.0.1:8765`, loads the estimator once, and
serializes inference with a lock. Check it from the `robomme` environment:

```bash
PYTHONPATH=src python -c \
  "from harness.grounding import F2GClient; print(F2GClient().health())"
```

If `harness` is not importable, run with `PYTHONPATH=src` or install this
repository in the environment.

Run the lightweight contract, transform, observation, and synthetic OpenCV
checks in the `robomme` environment:

```bash
PYTHONPATH=src python -m pytest \
  src/harness/tests/test_grounding_contracts.py \
  src/harness/tests/test_grounding_runtime.py -q
```

## Evaluation integration

The environment must be built with all four inputs enabled/present:

```python
env = builder.make_env_for_episode(
    episode_id,
    include_front_depth=True,
    include_front_camera_extrinsic=True,
    include_front_camera_intrinsic=True,
)
```

After the VLM returns one validated action and its top-level `grounding`:

```python
from harness.grounding import GroundingPipeline
from harness.grounding import camera_frame_from_robomme
from harness.grounding import depth_statistics

frame = camera_frame_from_robomme(
    obs,
    info,
    depth_scale=0.001,  # metres per raw unit; verify this once on the server
)
print(depth_statistics(frame))

grounded = GroundingPipeline().build_action_request(
    action_name=next_subgoal_key,
    grounding=vlm_output["grounding"],
    frame=frame,
    arguments=vlm_output.get("arguments", {}),
)
result = action_executor.execute(grounded.request)
```

For a tabletop camera, the reported `median_m` should be physically plausible
(usually around 0.5-1.5 m). Start with `0.001`: if the result is around
`0.0005`, raw depth was already in metres and the scale must be `1.0`. If a
trial with scale `1.0` reports hundreds of metres, raw depth is millimetres and
the scale must be `0.001`. Do not tune segmentation before this check passes.

`grounded.masks` contains the chosen masks and diagnostics for logging. Save
these during initial evaluation; a wrong mask can otherwise look like a pose
failure.

## Files

- `observation.py`: latest RoboMME RGB-D/camera-parameter adapter;
- `opencv_segmenter.py`: color+point and depth+point mask selection;
- `f2g_client.py`: compressed localhost request to F2G;
- `transforms.py`: `T_cam_world`/`T_cam_obj` to `T_world_obj` and quaternion;
- `grounder.py`: trusted catalog orchestration and `ActionRequest` creation;
- `scripts/f2g_pose_server.py`: process launched by the `f2g` environment.

The `grid_targets` detector is intentionally a first-pass HSV connected
component implementation. The pipeline reorders its F2G estimates by world
`y`, matching RouteStick's negative-to-positive `buttons_grid` construction.
Before using PatternLock/RouteStick for scoring, still confirm the candidate
count and ordering on saved masks; missed or merged circles shift every later
index.
