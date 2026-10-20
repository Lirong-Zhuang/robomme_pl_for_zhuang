# Versioned Manager and Reporter prompts

QwenVL Manager and Reporter prompts are selected by stable version names. The
same formatter is used during dataset construction and evaluation so prompt
text cannot silently drift between training and inference.

## Existing versions

Reporter:

- `temporal_v1`: the Trinity v2.1 temporal comparison prompt.
- `interaction_aware_v2`: the Trinity v2.2 interaction-aware single-view
  prompt retained for existing checkpoints.
- `dual_view_temporal_v3`: the current Trinity v2.2 default. It uses a rolling
  window of up to three Reporter-call observations, each ordered as overall
  image then wrist image, and compares them with an overall/wrist init pair.
  Eval deletes the temporary init-frame directory after each episode.

Reporter aliases:

- `trinity_v2.1` -> `temporal_v1`
- `trinity_v2.2` -> `dual_view_temporal_v3`
- `trinity_v2.2-interaction-aware` -> `interaction_aware_v2`
- `trinity_v2.2-dual-view` -> `dual_view_temporal_v3`

Manager:

- `qwenvl_v1`: the QwenVL Manager prompt shared by Trinity v2.1 and v2.2 and
  the current default.

The Manager registry currently applies only to QwenVL. Gemini, MemER, and QPA
retain their specialized prompt implementations.

## Build datasets

Build one Reporter dataset with the old prompt:

```bash
uv run python scripts/build_dataset.py \
  --dataset_type reporter_qwenvl \
  --reporter_prompt_version temporal_v1 \
  --reporter_history_size 7
```

Build an episode-disjoint Reporter train/test dataset with the v2.2 prompt:

```bash
uv run python scripts/build_trainset_testset.py \
  --dataset_type reporter_qwenvl \
  --reporter_prompt_version dual_view_temporal_v3 \
  --reporter_history_size 3
```

Build a QwenVL Manager dataset:

```bash
uv run python scripts/build_trainset_testset.py \
  --dataset_type manager_qwenvl \
  --manager_prompt_version qwenvl_v1
```

Each QwenVL Manager or Reporter output directory contains
`prompt_metadata.json`. Split builds also record the canonical version and
SHA-256 content hash in `split_manifest.json`.

## Run evaluation

For full environment evaluation, edit these values in `scripts/eval.sh`:

```bash
MANAGER_PROMPT_VERSION="qwenvl_v1"
REPORTER_PROMPT_VERSION="dual_view_temporal_v3"
```

Then run `bash scripts/eval.sh`. The resolved prompt versions and hashes are
written to `prompt_metadata.json` in each seed/repeat result directory. A
resume with different prompts is rejected; use a new run name or overwrite the
old result directory.

For offline Reporter evaluation:

```bash
python user_code/test_reporter.py \
  --reporter-prompt-version temporal_v1 \
  --result-name reporter-v1-prompt
```

This intentionally permits compatible cross-prompt experiments. The
`dual_view_temporal_v3` media layout is different from the older prompts, so it
must be used with a newly built dual-view dataset and a matching checkpoint.

## Add a prompt version

1. Add a new immutable module under `src/mme_vla_suite/prompts/reporter/` or
   `src/mme_vla_suite/prompts/manager/`.
2. Give it a unique canonical `VERSION` value and define all templates required
   by the corresponding existing module.
3. Import it in `src/mme_vla_suite/prompts/registry.py` and add one
   `ReporterPromptSpec` or `ManagerPromptSpec` entry.
4. Check the rendered semantic difference and placeholder count without adding
   a repository test file solely for feasibility validation.
5. Build a new dataset. Never edit a prompt module already used by a recorded
   experiment; doing so changes its hash while leaving its name unchanged.

Reporter user prompts are dynamic because the number of `<image>` placeholders
must equal the prompt version's declared media count. For the default dual-view
prompt this is `2 + 2 * observation_count`; the legacy single-view prompts use
`1 + observation_count`. Manager prompt versions must define both simple and
grounded system prompts, initial-turn formatting, follow-up formatting, and all
three Reporter feedback states (`true`, `false`, and unavailable).

## Rename a version or change the default

Canonical names are experiment identifiers and should normally not be renamed.
To expose a new invocation name without breaking old commands, add an entry to
`_REPORTER_ALIASES` or `_MANAGER_ALIASES` in `prompts/registry.py`.

If a canonical rename is unavoidable:

1. Change the module's `VERSION` and registry key.
2. Map the old name to the new name in the relevant alias dictionary.
3. Keep existing manifests and result metadata unchanged; they describe the
   historical resolved name.

To change the default for commands that omit the switch, update
`DEFAULT_REPORTER_PROMPT_VERSION` or `DEFAULT_MANAGER_PROMPT_VERSION` in
`prompts/registry.py`. Explicit version arguments always take precedence.
