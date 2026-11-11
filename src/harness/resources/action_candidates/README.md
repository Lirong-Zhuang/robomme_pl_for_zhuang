# Action candidate resources

Store task-scoped, machine-readable action manifests in this directory. Prefer
JSON files named with snake_case task categories, for example:

- `bin_fill.json`
- `move_cube.json`
- `pick_highlight.json`

Each entry should reference an existing trusted action implementation and declare
its stable action name, purpose, input contract, preconditions, observable effects,
and whether visual grounding is required. Do not put executable Python or generated
task plans in these files.

The init planner receives exactly one of these files as
`action_candidates_path`. Keeping the manifest task-scoped prevents the VLM from
selecting irrelevant actions.
