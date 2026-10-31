# Qwen 2.7B feasibility test

This directory is an isolated offline experiment. It does not call RoboMME or
MPLib actions.

## Files

- `prompts/base.py`: shared ordinary, non-grounded initial/follow-up prompt
  formats.
- `prompts/binfill.py`: BinFill-specific notes, possible subgoals, examples,
  three MPLib wrapper names, and English action notes.
- `prompts/__init__.py`: task prompt registry used by the dataset builder.
- `build_test_dataset.py`: selects an exact number of episodes per task and
  creates one joint dual-view JSONL dataset.
- `run_test.py`: runs the joint test with a local Qwen-VL checkpoint through
  ms-swift and writes per-sample predictions plus a summary.
- `runtime_metrics.py`: records single-step wall-clock latency and best-effort
  NVIDIA GPU memory snapshots through NVML.

## Build 10 or 20 episodes

Run from the repository root. `--episode-count` is applied independently to
each selected task; no percentage split is calculated.

```bash
python user_code/test_2.7B/build_test_dataset.py \
  --raw-data-path data/h5_data/merged \
  --dataset-name binfill_1 \
  --episode-count 1
```

The default output directory is `data/test_2.7B/<dataset-name>`, so the command above
writes `data/test_2.7B/binfill_10`.

Change `--episode-count 10` to `--episode-count 20` for twenty episodes. To
use known IDs instead of random fixed-count selection:

```bash
python user_code/test_2.7B/build_test_dataset.py \
  --raw-data-path data/h5_data/merged \
  --dataset-name binfill_explicit \
  --episode-ids 6 16 22 36 42 64 66 69 75 89
```

The output contains:

```text
images/overall/
images/wrist/
subgoal_action_test.jsonl
manifest.json
```

Each selected episode contributes two kinds of calls:

- One `initial` call that uses the initial overall/wrist pair and predicts the
  complete paired subgoal/action sequence, its first subgoal, and its matching
  wrapper action.
- Multiple `followup` calls that compare the subgoal-init pair with recent
  overall/wrist pairs, decide whether the current subgoal is complete, and
  retain or advance both the subgoal and action in one response.

The builder defaults to `--tasks BinFill`. `--tasks` remains available for
explicit task selection. At present only BinFill has a task-specific prompt;
add another module and register it in `prompts/__init__.py` before selecting a
different task.

At terminal completion the expected output is `next_subgoal="STOP"` and
`action_name=null`; the model must not select another robot action.

## Validate without loading the model

```bash
python user_code/test_2.7B/run_test.py \
  --dataset-name binfill_10 \
  --result-name binfill_10_dry_run \
  --dry-run
```

## Run joint subgoal and action prediction

Activate the server environment containing `ms-swift`, then run:

```bash
python user_code/test_2.7B/run_test.py \
  --dataset-name binfill_1 \
  --result-name test_1.1 \
  --cuda-visible-devices 0 \
  --attn-impl flash_attention_2
```

The default local model path is:

```text
/data/public/qwen/Qwen3.8-27B-FP8
```

Use `--model-path` or `QWEN_MODEL_PATH` to override it.

## Low-latency inference settings

The runner explicitly builds the ms-swift template with
`enable_thinking=False` and `preserve_thinking=False`. It verifies the setting
before starting the test, so Qwen3.8 cannot silently fall back to its hybrid
thinking mode. The prompt also requests JSON without analysis, reasoning, or
think tags.

The default latency-oriented configuration is:

```text
attention backend: flash_attention_2
temperature: 0.0
top_k: 1
top_p: 1.0
repetition_penalty: 1.0
max output tokens: 128
maximum tokens per image: 256
batch size: 1
CUDA memory caching: enabled
```

`batch_size=1` measures true request latency rather than maximizing aggregate
throughput. CUDA allocation caching remains enabled because disabling it is a
debugging option that increases allocation overhead. If the server environment
does not contain FlashAttention, use `--attn-impl sdpa`; this is more compatible
but may be slower.

Every `predictions.jsonl` row and the final `summary.json` record these values
under `inference_config`, including `enable_thinking=false`, so the effective
test configuration can be audited afterward.

`--result-name` is required for every launch. The launcher writes all test
results under `runs/test_2.7B/<result-name>/`:

```text
predictions.jsonl
buffered_completion_episodes.jsonl
summary.json
```

`predictions.jsonl` is the per-step input/output file. Every row contains the
exact input messages sent to the model, input image paths, request metadata,
expected output, parsed prediction, raw model response, strict scores, and
timing. Its main structure is:

```json
{
  "test_step": 1,
  "metadata": {
    "task": "BinFill",
    "episode": 6,
    "current_step": 42,
    "call_type": "followup"
  },
  "input": {
    "messages": [{"role": "user", "content": "...<image>..."}],
    "images": [".../overall.png", ".../wrist.png"]
  },
  "expected": {
    "completed": false,
    "next_subgoal": "pick up the cube",
    "action_name": "solve_pickup"
  },
  "output": {
    "raw_response": "{...}",
    "parsed": {
      "completed": false,
      "next_subgoal": "pick up the cube",
      "action_name": "solve_pickup"
    }
  },
  "timing": {
    "vlm_response_ms": 842.617,
    "image_count": 8,
    "measurement_scope": "InferRequest construction with image paths through engine.infer returning output"
  }
}
```

The timer starts immediately before constructing the image-bearing
`InferRequest` and stops when `engine.infer` returns. It therefore includes
request construction, image loading/preprocessing performed by the engine,
prefill, decoding, and response return. `--batch-size` must be `1`, so the
reported number is the real wall-clock response time for that step rather
than a batch average.

## Strict and buffered accuracy

The original strict per-row metrics remain in `summary.json`:

- `completed_accuracy`
- `subgoal_accuracy`
- `action_accuracy`
- `joint_accuracy`
- `pair_consistency`

`buffered_completion` additionally calls the same causal scoring function as
the existing Reporter test. Its defaults are:

```text
early_tolerance_calls = 1
full_credit_delay_calls = 1
maximum_delay_calls = 2
reporter_debounce = true
```

Thus a completion event may be one Reporter call early; a delay of one call
still has full credit; a delay of two calls completes with a warning. A more
premature event, a missing event, or a later event stops progress for that
episode. Continuous `completed=true` predictions use the same Reporter
debounce/retrigger rule. Override the defaults with:

```bash
python user_code/test_2.7B/run_test.py \
  --dataset-name binfill_10 \
  --result-name binfill_10_joint \
  --early-tolerance-calls 1 \
  --full-credit-delay-calls 1 \
  --maximum-delay-calls 2
```

The buffer is applied only to the timing of the follow-up `completed` event.
`next_subgoal` and `action_name` continue to use strict text accuracy, which
prevents a temporally tolerated completion from hiding a wrong next decision.
The complete per-episode transition trace is stored in
`buffered_completion_episodes.jsonl`.

This fixed JSONL dataset advances its prompt using the ground-truth segment.
Consequently the buffered score is a useful offline temporal diagnostic, but
it is not identical to a live prediction-driven rollout after a late or early
transition. Use strict subgoal/action metrics for this feasibility test; add a
prediction-driven evaluator before treating buffered completion as a robot
rollout success rate.

## GPU and timing summary

`summary.json` contains a `gpu_and_timing` block with this shape:

```json
{
  "gpu_and_timing": {
    "gpu": {
      "physical_gpu_index": 0,
      "gpu_name": "NVIDIA ...",
      "total_memory_gb": 80.0,
      "global_used_before_model_load_gb": 11.72,
      "other_processes_before_model_load_gb": 11.72,
      "current_process_after_model_load_gb": 27.83,
      "test_increment_after_model_load_gb": 27.83,
      "peak_current_process_during_test_gb": 29.3,
      "test_peak_increment_over_baseline_gb": 29.3
    },
    "timing": {
      "steps": 100,
      "model_load_seconds": 21.4,
      "average_step_vlm_response_ms": 842.6,
      "median_step_vlm_response_ms": 810.2,
      "p95_step_vlm_response_ms": 1104.8,
      "min_step_vlm_response_ms": 701.3,
      "max_step_vlm_response_ms": 1340.5
    }
  }
}
```

All GPU-memory values use GB, with `1 GB = 1024^3 bytes`. The baseline snapshot
is taken after `CUDA_VISIBLE_DEVICES` is set but before ms-swift imports or
loads the model. `other_processes_before_model_load_gb` is therefore the best
estimate of memory already occupied by other users.
`test_increment_after_model_load_gb` and
`test_peak_increment_over_baseline_gb` are global-memory deltas from that
baseline; `peak_current_process_during_test_gb` is the NVML attribution for
this Python process. If the NVIDIA driver or `pynvml` is unavailable, the test
still runs and the GPU block records `available=false` plus the error.

## Summarize an unfinished framework timing run

To aggregate the first 30 completed BinFill episode timing files without
waiting for the full evaluation run:

```bash
python user_code/test_2.7B/summarize_partial_timing.py
```

The default source directory is:

```text
runs/evaluation/symbolic-simple-subgoal/trinity_v2.2.2/seed0/repeat1/BinFill/timing_logs
```

The default result is `temporary_summary_30episodes.json` in that directory.
Component averages are weighted over all calls using
`sum(total_ms) / sum(call_count)`. To use another partial count or directory:

```bash
python user_code/test_2.7B/summarize_partial_timing.py \
  --num-episodes 20 \
  --timing-dir runs/evaluation/other_run/seed0/repeat1/BinFill/timing_logs
```

