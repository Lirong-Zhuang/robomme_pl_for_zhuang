# Task-config example resources

Store reusable `task_config.v1` examples and defaults in this directory. The
current test fixtures are complete JSON examples:

- `binfill.json`
- `movecube.json`
- `pickhighlight.json`

Each file is one complete worked example that may provide structure and writing
style to the init VLM. An example must not contain a concrete episode goal, current
observations, or generated runtime state.

These task-specific files are gold/reference outputs and end-to-end smoke-test
fixtures. Do not pass the matching task file to the init VLM when measuring whether
it can infer video or keyframe requirements from the minimal task description; that
would reveal the expected answer. For that experiment, `task_config_example_path`
should point to an example from an unrelated task, or the no-example prompt variant
should be selected with `use_task_config_example=False`.
