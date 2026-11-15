# Manual OpenCV mask test resources

Place the source image to inspect in this directory. By default the manual test
expects:

```text
current_front_overall.png
```

The defaults can be edited at the top of
`src/harness/tests/test_opencv_manual_mask.py`, or overridden when running the
test:

```bash
AGENTIC_MASK_IMAGE=current_front_overall.png \
AGENTIC_MASK_COLOR=red \
AGENTIC_MASK_X=120 \
AGENTIC_MASK_Y=110 \
AGENTIC_MASK_OUTPUT_NAME=binfill_red_test \
python src/harness/tests/test_opencv_manual_mask.py
```

Direct Python execution requires only NumPy and OpenCV from the `robomme`
environment; pytest is not required. If pytest is later installed, set
`RUN_AGENTIC_OPENCV_MASK_TEST=1` to enable the optional pytest wrapper.

Optional `AGENTIC_MASK_RGB_DISTANCE` overrides the default RGB-distance
threshold. `AGENTIC_MASK_OUTPUT_NAME` controls the common filename prefix for
all generated artifacts. Outputs are written to:

```text
runs/agentic/test/grounding/
```

The generated artifacts are the raw RGB-distance mask, the nearest selected
component mask, a green mask overlay with the requested point rendered as the
same filled yellow dot used by Oracle Plus, and a JSON diagnostics file.
