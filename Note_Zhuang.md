
## Automat

change line 25-84 of file scripts/eval.sh to change the setup

```bash
bash scripts/eval.sh
```



## Manuel

This project uses two separate Python environments.

### 1. Policy Learning / uv Environment

Use this environment for the VLA policy learning code and policy server.

Activate:

```bash
source .venv/bin/activate
```

Deactivate:

```bash
deactivate
```

Typical usage:

```bash
uv run scripts/serve_policy.py ...
```

### 2. RoboMME Simulator / conda Environment

Use this environment for the RoboMME simulator, `simple_test.py`, and `eval.py`.

Activate:

```bash
conda activate robomme
```

Deactivate:

```bash
conda deactivate
```

Typical usage:

```bash
python examples/robomme/simple_test.py
python examples/robomme/eval.py ...
```

## Run a eval to observe the response

change line 47-64 of file examples/robimme/eval.py to change the setup

### Terminal 1 with uv env(openpi)

```bash
CUDA_VISIBLE_DEVICES=0 uv run scripts/serve_policy.py
```

### Terminal 2 with conda env(robomme)

```bash
CUDA_VISIBLE_DEVICES=0 python examples/robomme/eval.py
```
