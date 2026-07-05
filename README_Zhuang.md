

## Virtual Environments

This project uses two separate Python environments.

### 1. Policy Learning / uv Environment

Use this environment for the VLA policy learning code and policy server.

Activate:

```bash
cd ~/tianyaoTong/lirongZhuang/robomme_pl_for_zhuang
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

