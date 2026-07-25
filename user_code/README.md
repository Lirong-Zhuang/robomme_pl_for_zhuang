# Personal dataset tools

Inspect one RoboMME HDF5 timestep:

```bash
uv run python user_code/inspect_h5_timestep.py \
  data/robomme_data_h5/record_dataset_BinFill.h5 \
  --episode 1 \
  --timestep 0 \
  --output-dir runs/h5_inspection
```

The command prints episode setup and timestep values, then creates:

```text
runs/h5_inspection/episode_0/timestep_0/
├── report.txt
├── summary.json
├── images/
│   └── *.png
└── arrays/
    └── *.npy
```

RGB arrays are saved directly as PNG. Depth and other image-shaped numerical
arrays are min-max normalized for PNG preview, while their lossless raw values
are also saved as NPY.

Export a complete episode:

```bash
uv run python user_code/inspect_h5_timestep.py \
  data/robomme_data_h5/record_dataset_BinFill.h5 \
  --episode 1 \
  --all-timesteps \
  --output-dir runs/h5_inspection
```
