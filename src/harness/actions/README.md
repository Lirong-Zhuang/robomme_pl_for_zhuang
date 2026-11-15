# Harness action layer

This package executes stable names from `resources/action_candidates/action_catalog.json`.
Perception/grounding must supply `ObjectEstimate` values; action implementations must not
look up semantic targets such as `env.unwrapped.cube` or accept SAPIEN actors as targets.

## Oracle Plus parity contract

This is a port of the action path used by Oracle Plus, not a new action-policy design.
For each action, `oracle_reference.py` names the exact RoboMME/Oracle source that acts as
its behavioral specification. Implementations must initially preserve:

- waypoint order and numerical offsets/heights;
- TCP orientation construction and quaternion multiplication order;
- gripper open/close timing;
- screw planning, RRT* fallback and retry behavior from `OraclePlusPlannerRuntime`;
- bounded dense-step collection, early episode stop, and final status evaluation;
- task arguments such as `direction`, `radius`, `release_z` and absolute timestep;
- task-specific legacy `obj_flag` values derived from trusted black/white peg-end
  pickup state rather than predicted by a later VLM call.

The intended replacement boundary is limited to target data. Reads such as `obj.pose`,
`target.pose`, `get_actor_obb(obj)`, `env.box` and task-owned actor lists are replaced by
grounded `ObjectEstimate` geometry produced from OpenCV/F2G. Robot state, environment
stepping and MPLib execution still come from the Oracle Plus runtime path.

The in-task VLM never produces an `ObjectEstimate`. It may provide only one
current-image `[x, y]` point per required semantic target and the catalog-declared
color hint. OpenCV expands those hints into single-object masks; F2G is the sole
producer of metric position, orientation, pose, and size before the result crosses
this package boundary.

## File responsibilities and fill order

1. `types.py`: canonical world/base-frame geometry and request/result contracts. Change this
   only when the perception-to-action contract changes.
2. `oracle_reference.py`: exact source function for every catalog action. Check this before
   filling any action body.
3. `geometry.py`: port only the source function's quaternion/OBB math here and test it
   without a simulator. Do not introduce a new motion policy.
4. `pick_place.py`: Oracle-style functions retaining `env, planner`, with actor parameters
   replaced by explicit `obj_pose`, `obj_size`, or `target_pose` keyword arguments.
5. `button.py`: port Oracle's fixed ready/contact/retreat sequence, taking only the button
   pose from the estimate.
6. `push.py`, `peg.py`, `trajectory.py`: port their Oracle sources after the basic vertical
   slice works.
7. `temporal.py`: contains the Oracle absolute-timestep implementation.
8. `registry.py`: deterministically unpacks `ObjectEstimate` into the explicit function
   parameters; it must not implement motion logic.
9. `executor.py`: invokes the selected function inside the existing
   `OraclePlusPlannerRuntime.execute()` collector.

Implement pickup and placement first. Suggested BinFill target roles are
`object` for the cube and `container` for the bin/opening.

Each function keeps the corresponding Oracle Planner motion body and replaces only actor
geometry reads with explicit pose/size inputs. The function returns the same raw planner
value as Oracle; `ActionExecutor` converts the runtime execution result into `ActionResult`.

## Canonical execution path

```text
TaskMemory.current_execution()
  -> semantic argument resolver scans rendered subgoal word tokens
  -> perception/grounding binds semantic target roles
  -> ActionRequest
  -> ActionExecutor
  -> OraclePlusPlannerRuntime.execute
  -> registered Oracle-style function(env, planner, **explicit_geometry)
```

Before implementing `solve_pickup`, extract and preserve these Oracle parameters explicitly:

- approach clearance;
- finger depth and collision margin;
- how `get_actor_obb` and `compute_grasp_info_by_obb` select the closing axis;
- grasp/contact height;
- post-grasp lift height.

Do not tune these values while doing the first port. First obtain behavior parity with the
Oracle source; parameter cleanup or improvements should be a separate, testable change.

For `solve_pattern_direction`, `target_sets["targets"]` contains all perceived grid targets.
For `solve_swingonto_with_direction`, that tuple must preserve the original `buttons_grid`
index order because Oracle RouteStick uses indices `(0, 2, 4, 6, 8)` to define allowed
targets and robot-left/robot-right neighbors.
