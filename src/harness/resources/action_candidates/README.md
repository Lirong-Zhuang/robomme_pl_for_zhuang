# Action candidate resources

Store machine-readable action manifests in this directory. The current test uses
one global `action_catalog.json` containing every trusted candidate exposed to the
init planner; actions are not grouped or labeled by task.

For the initial test, each entry contains only the stable action name and a concise
description. Do not put executable Python, task labels, or generated task plans in
this file.

The init planner receives this file as `action_candidates_path` and must infer the
relevant subset from the task description and action semantics.
