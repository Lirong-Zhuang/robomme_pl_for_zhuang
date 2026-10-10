# Versioned Manager and Reporter prompts

QwenVL Manager and Reporter prompts are selected by stable version names. The
same formatter is used during dataset construction and evaluation so prompt
text cannot silently drift between training and inference.

## Existing versions

Reporter:

- `temporal_v1`: the Trinity v2.1 temporal comparison prompt.
- `interaction_aware_v2`: the Trinity v2.2 interaction-aware prompt and the
  current default.

Reporter aliases retained for historical commands:

- `trinity_v2.1` -> `temporal_v1`
- `trinity_v2.2` -> `interaction_aware_v2`
- `trinity_v2.2-interaction-aware` -> `interaction_aware_v2`

Manager:

- `qwenvl_v1`: the QwenVL Manager prompt shared by Trinity v2.1 and v2.2 and
  the current default.
- `qwenvl_interaction_verify_v2`: a prompt-mismatch evaluation variant that
  preserves the `qwenvl_v1` system and initial prompts, adds a short
  Reporter-success verification gate to follow-up prompts, and injects the
  detailed interaction rules only when Reporter returns `true`. It verifies
  subgoal-specific relations for pick-up, put-into-bin, and button subgoals
  before advancing. On that `true` branch, the Manager returns a compact JSON
  record containing the checked interaction pairs, the verification result,
  and the selected subgoal; evaluation logging records the JSON while passing
  only its `subgoal` field to the Executer. The Manager model itself infers the
  relevant interaction pairs and decides whether the visual success criteria
  are satisfied. Interaction pairs describe the relations being checked, so
  the Manager must still return the required pairs when its visual decision is
  false; Python does not classify subgoals or validate relation names.
  Reporter advances its init frame only when the Manager explicitly returns
  `verification_passed=true`. A false, missing, or malformed Manager decision
  keeps both the current subgoal and the existing init frame.
  The `false` and missing-result branches retain the original plain-subgoal
  output behavior.

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
  --reporter_prompt_version interaction_aware_v2 \
  --reporter_history_size 7
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
REPORTER_PROMPT_VERSION="interaction_aware_v2"
```

To run the interaction-verification ablation without retraining, set:

```bash
MANAGER_PROMPT_VERSION="qwenvl_interaction_verify_v2"
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

This intentionally permits cross-prompt experiments, such as evaluating an
adapter trained with `temporal_v1` using `interaction_aware_v2`. Such a result
is a prompt-mismatch ablation, not a like-for-like checkpoint comparison.

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
must equal `1 + observation_count`. New Reporter formatters must preserve that
invariant. Manager prompt versions must define both simple and grounded system
prompts, initial-turn formatting, follow-up formatting, and all three Reporter
feedback states (`true`, `false`, and unavailable).

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
