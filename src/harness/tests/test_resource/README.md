# Manual OpenCV mask test resources

Place the source image to inspect in this directory. The configured default is:

```text
red_cube_only.png
```

Edit the configuration block at the top of
`src/harness/tests/test_opencv_manual_mask.py`, then run:

```bash
python src/harness/tests/test_opencv_manual_mask.py
```

Direct Python execution requires only NumPy and OpenCV from the `robomme`
environment; pytest is not required. If pytest is later installed, set
`RUN_AGENTIC_OPENCV_MASK_TEST=1` to enable the optional pytest wrapper.

`OUTPUT_NAME` controls the common filename prefix for generated artifacts. The
program derives an OpenCV HSV prototype from the predefined RGB color, then
uses circular hue distance plus saturation/value gates so shaded parts of the
same object remain in the mask. Outputs are written to:

```text
runs/tests/grounding/
```

The manual mask program writes only `<output_name>_mask_overlay.png`. Its final
selected mask is painted pure black and the requested point is rendered as the
same filled yellow dot used by Oracle Plus. Detailed diagnostics are printed to
the terminal instead of creating additional files.

## Full F2G grounding test

The full grounding test additionally requires `grounding_input.npz`, copied
from an Agentic Eval `grounding/ep_<id>/call_<id>/` directory. This file contains
the aligned depth, intrinsic, world-to-camera matrix, and depth scale required
by F2G. Keep an F2G service running in the separate `f2g` environment, then run
`src/harness/tests/test_grounding.py` in the `robomme` environment. All image,
point, mask, output, NPZ, and F2G endpoint settings are in the configuration
block near the top of that file, so the test itself is launched with one line:

```bash
bash scripts/serve_f2g_for_test.sh  # terminal A
python src/harness/tests/test_grounding.py
```

It writes only:

```text
runs/tests/grounding/<output_name>_mask_overlay.png
runs/tests/grounding/<output_name>_grounding.png
```
