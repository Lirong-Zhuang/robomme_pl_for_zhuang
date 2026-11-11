# Task-config package resources

Store reusable `task_config.v1` examples and defaults in this directory. Packages
should be organized by task family rather than by episode, for example:

- `counting.json` for tasks such as BinFill
- `imitation.json` for tasks such as MoveCube
- `reference.json` for tasks such as PickHighlight

A package may contain a base config, one or more worked examples, recommended
verification defaults, prompt templates, and optional keyframe-policy examples. It
must not contain a concrete episode goal, current observations, or generated runtime
state.

The init planner receives the selected family package as
`task_config_package_path` and adapts it to the registered task description and
the task-scoped action candidates.
