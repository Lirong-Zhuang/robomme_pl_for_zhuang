# Task-config examples organized by information source

The active example set contains three complete `task_config.v1` examples. They are
named by the information sources needed to turn an episode task goal into a
concrete, ordered subgoal sequence, rather than by task name:

- `task_goal_only.json`
- `task_goal_plus_video_subgoal_list.json`
- `task_goal_plus_keyframes.json`

The init prompt receives the path to this Markdown file, not a preselected JSON
path. The init-planning VLM must read this index first, apply the source-profile
selection rule below to the current task, and then read exactly one sibling JSON
example for the selected profile.

Each file is a worked structural and writing-style example. The init planner must
still infer a new config for the current task; it must not copy an example as the
current task's config or treat an example as an episode plan. An example must not
contain a concrete episode goal, current observations, or generated runtime state.
The represented tasks merely demonstrate three metadata-acquisition patterns. When
the current task is a different task, every task-specific subgoal, action mapping,
argument, ordering rule, acceptance criterion, object role, note, keyframe event,
and runtime prompt must be generated anew from the current task inputs. Renaming an
example's task or objects is not an acceptable generated config.

Every example uses the same two-prompt runtime interface:

- `first_planning` is called once before any subgoal has executed. It performs no
  verification. It builds the concrete sequence currently inferable from the
  available sources and returns its first `next_subgoal_key`, instantiated subgoal
  text, and arguments. Every sequence entry uses the same three fields.
- `followup_planning` is called after a subgoal execution attempt. In one response
  it verifies the current subgoal as `completed`, `incomplete`, or `failed`, gives
  concise observable evidence, and selects the next key, text, and arguments. An
  incomplete subgoal remains active; only a completed subgoal advances.

Each top-level subgoal has a unique stable `subgoal_key`. The VLM never outputs an
action name. The harness resolves `subgoal_key` to the task config's trusted action
and acceptance criteria, validates the instantiated text, and rejects unknown or
mismatched keys.

Object identity and operational notes are intentionally separate:

- `object_roles` is a minimal runtime glossary. Add an entry only when omitting its
  visible identity could cause the VLM to confuse interactive objects or choose,
  execute, or verify the wrong object. Each entry contains exactly `role` and
  `identity`. Generic identities already unambiguous from `task_goal`, observations,
  action names, and subgoal text may be omitted.
- `task_notes` contains only non-obvious rules that can change planning, execution,
  memory maintenance, or verification. It is not an object glossary and must not
  restate the task description or facts reliably available from runtime inputs.
- Applicable entries from both fields are inlined into the generated first and
  follow-up system prompts. Neither field is passed as a runtime placeholder.

Runtime memory is split into two files inside each episode directory:

- `task_state.json` is a compact, atomically replaced working state passed to normal
  follow-up planning calls.
- `task_events.jsonl` is an append-only detailed event log read only for audit,
  recovery, or explicit history lookup.

`harness.task_memory.TaskMemory` owns both files and provides first-plan commit,
action-event recording, combined verification/transition commit, current action and
acceptance lookup, compact follow-up input construction, and filtered event queries.
Video-derived facts and keyframe descriptions declare a `task_state_field` under
`persistent_facts`; `set_task_state_field()` writes that compact value and appends
the same update to the detailed event log.

Only configs with a non-empty `keyframes` list add `keyframe_save` to every first
and follow-up planner result. Its value is either `null` or an exact observation ID
supplied by the execution loop. `TaskMemory.save_keyframe_selection()` validates
that ID against the loop's trusted ID-to-image-path mapping and stores the resolved
image reference at the configured `task_state_field`. It never accepts a VLM-invented
path and does not copy or move the underlying image file.

Keyframe policies are not required to be minimal. Generate every justified policy
needed to preserve task-relevant information across all later environment changes.
Before finalizing them, check that `task_goal`, the retained evidence, and ordinary
later observations can jointly recover every target identity and cross-time
correspondence needed by later planning. A pre-change snapshot is insufficient when
visually similar objects subsequently move or swap; use a sequence policy or
additional event policies that preserve the identity transformation.

`capture_mode: "single"` uses `frame_count: 1` and stops after one save.
`capture_mode: "sequence"` uses `frame_count` greater than one as a safety upper
bound, requires an observable `stop_when`, and may append one matching observation
on each planning call until `stop_when` is satisfied or the upper bound is reached.
The first observation satisfying `stop_when` is retained as the terminal frame when
capacity remains. Existing partial sequence evidence does not close an otherwise
active policy. Every keyframe policy sets `save_description: true`, and every
retained observation must have the description requested by
`description_instruction`.

Those configs also add `{keyframe_policy}` to their user prompts. TaskMemory builds
this runtime object directly from the applicable `task_config.keyframes[index]`
entry and adds a `task_config_ref` such as `task_config.keyframes[0]`. Consequently,
the exact `save_when`, view, capture, description, and destination settings are
visible in every relevant user message without relying on a system-prompt paraphrase.
They additionally receive `{saved_keyframe}` explicitly: `null` before capture. A
single policy then receives one retained image-placeholder reference, trusted image
path, observation ID, and description. A sequence policy receives its ordered saved
observations and descriptions together with `saved_count`, the frame-count upper
bound, and the stop condition. Video-list configs similarly receive the locked ordered list as
`{video_subgoal_list}` on every follow-up call. Their system prompts mark these
objects, together with `task_goal`, as mandatory subgoal-planning references.

The JSON files store prompt newlines as `\n`, as required by strict JSON. To view
one prompt as normal formatted text without changing what the VLM receives, run:

```bash
jq -r '.vlm_prompts.first_planning.system_prompt' \
  src/harness/resources/task_config_examples/task_goal_only.json
```

## Source-profile selection rule

Before choosing an example, determine whether a future concrete `task_goal`, in the
format illustrated by the representative task-goal examples, contains enough
information to uniquely construct every concrete subgoal. Test all of the following:

- the required `subgoal_key` or semantic subgoal type;
- its trusted action mapping;
- the number and order of subgoals;
- every argument value needed to instantiate the subgoal; and
- the semantic target identity or cross-time object correspondence needed to choose
  the correct currently visible object.

Ordinary visual localization of an object already named unambiguously by
`task_goal`, such as finding the currently visible green cube, is not missing
metadata and does not by itself justify video or keyframe memory. A target identity
or correspondence is missing metadata when it cannot be recovered from the current
goal and current observations alone, for example which visually similar container
previously covered a requested cube.

1. Choose `task_goal_only.json` when `task_goal`, together with ordinary current
   observations for live localization, uniquely supplies every item above.
2. Choose `task_goal_plus_video_subgoal_list.json` when an initial demonstration
   video supplies any persistent fact that the text goal leaves unspecified, such
   as strategy, subgoal order or count, argument values, target identities, or
   cross-time object correspondences. Encode those facts in the concrete ordered
   subgoal list and retain that list for later planning calls.
3. Choose `task_goal_plus_keyframes.json` when task-relevant visual information
   needed for any item above appears transiently during execution, is not recoverable
   from `task_goal` and later observations, and must be captured before it disappears
   for later planning or verification.

Do not select an additional source merely because the task mentions video, memory,
or visual observations. Select it only when omitting that source would leave the
planner unable to construct the correct concrete subgoal sequence.

## Why the current examples use these profiles

### `task_goal_only.json` — represented by BinFill

BinFill task goals explicitly state the requested cube colors, their counts, and
that the button must be pressed after the placements. Those values are sufficient
to expand the goal into alternating pickup and bin-placement subgoals followed by
one button-press subgoal. No demonstration-derived strategy or transient visual cue
is required to decide the subgoal sequence.

### `task_goal_plus_video_subgoal_list.json` — represented by MoveCube

The MoveCube task goal asks the robot to reproduce the manner shown previously but
does not say whether that manner was tool hooking, closed-gripper pushing, or
pick-and-place. This missing choice changes both the action names and the ordered
subgoals. The planner must therefore inspect the initial demonstration, extract one
ordered strategy-specific subgoal list, and retain that list for execution.

Video-derived metadata need not change the action type or order to require this
profile. For example, VideoUnmask already implies a pickup/put-down structure, but
the initial demonstration supplies the container-to-cube correspondence needed to
instantiate each pickup with the correct target container. That correspondence must
therefore be encoded in and retained with the video-derived subgoal list.

### `task_goal_plus_keyframes.json` — represented by ButtonUnmaskSwap

ButtonUnmaskSwap requires two kinds of eval-time visual memory. One transient view
preserves the colored-cube positions before visually similar containers hide them.
The containers then swap while the cubes remain hidden, so an ordered sequence must
propagate each initial identity to its final container position. The worked config
therefore demonstrates both a single policy and a sequence policy under the same
`task_goal + keyframes` source profile.

The represented config demonstrates the full transport mechanism, not a mandatory
policy count. A new task retains only the policies justified by its own
information-loss process: one sufficient transient state may use only a single
policy, while later identity-changing motion may additionally require a sequence or
other event policies.

These files are also gold/reference outputs and end-to-end smoke-test fixtures. For
an inference experiment, select an example by source profile rather than by matching
the current task name, or use the no-example prompt variant with
`use_task_config_example=False`.
