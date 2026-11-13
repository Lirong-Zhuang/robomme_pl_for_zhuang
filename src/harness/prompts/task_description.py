"""Minimal paper-style task descriptions used by init planning."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TaskDescription:
    """Task summary plus separately typed identity and operational knowledge."""

    task_name: str
    summary: str
    task_goal_examples: tuple[str, ...]
    object_roles: tuple[str, ...] = ()
    operational_notes: tuple[str, ...] = ()

    def to_prompt_text(self) -> str:
        """Return only the paper-style summary shown to the init-planning VLM."""

        return self.summary


BINFILL_DESCRIPTION = TaskDescription(
    task_name="BinFill",
    summary=(
        "The BinFill task evaluates sequential decisionmaking and temporal memory. "
        "The robot must place a specified number of colored cubes into a bin and "
        "then press a button to terminate the episode. Success requires accurate "
        "counting and timely termination."
    ),
    task_goal_examples=(
        "Put one blue cube into the bin and press the button to stop.",
        (
            "Put one blue cube and two green cubes into the bin and press the "
            "button to stop."
        ),
        (
            "Put one blue cube, one green cube, and two red cubes into the bin "
            "and press the button to stop."
        ),
    ),
    object_roles=(
        (
            "Object-role identities: the manipulated objects are red, blue, or "
            "green cubes; the destination bin is a box with black holes on top; "
            "and the button terminates the episode after all placements."
        ),
    ),
    operational_notes=(
        (
            "Episode modes: (1) Static—all cubes are present at the beginning of "
            "the episode; (2) Streaming—cubes appear incrementally over time in a "
            "random order."
        ),
    ),
)


STOP_CUBE_DESCRIPTION = TaskDescription(
    task_name="StopCube",
    summary=(
        "The StopCube task challenges the robot’s temporal memory in a dynamic "
        "environment. The robot must monitor a cube that continuously oscillates "
        "across a target following a predefined line and press a button to stop "
        "exactly when the cube reaches the target at a specified ordinal occurrence "
        "(e.g., the third visit). Successful completion requires combining continuous "
        "visual tracking with discrete event counting to identify the correct "
        "occurrence and execute the stopping action with high temporal precision."
    ),
    task_goal_examples=(
        "Press the button to stop the cube exactly at the target on its third visit.",
    ),
    object_roles=(
        (
            "Object-role identities: the target is the purple circular disk, the "
            "moving object is the oscillating cube, and the button stops the cube."
        ),
    ),
    operational_notes=(
        (
            "Oscillating cube: the cube moves back and forth between start and end "
            "positions along a straight line."
        ),
        (
            "Static behavior: position the end-effector over the button and remain "
            "static, hovering until the correct timing."
        ),
        (
            "Immediate stop: pressing the button stops the cube instantly. Press at "
            "the exact moment the cube reaches the target zone in the specified "
            "cycle, accounting for the motion delay from hovering to the button."
        ),
    ),
)


PICK_XTIMES_DESCRIPTION = TaskDescription(
    task_name="PickXtimes",
    summary=(
        "The PickXTimes task evaluates the robot’s ability to perform iterative "
        "pick-and-place actions with temporal memory. The robot repeatedly picks up "
        "a cube of a specified color, places it on a target, and completes an exact "
        "number of cycles before pressing a button to terminate the task."
    ),
    task_goal_examples=(
        (
            "Pick up the blue cube and place it on the target, then press the "
            "button to stop."
        ),
        (
            "Pick up the blue cube and place it on the target, repeating this "
            "pick-and-place action three times, then press the button to stop."
        ),
    ),
    object_roles=(
        (
            "Object-role identities: the manipulated object is the requested red, "
            "blue, or green cube; the placement target is the purple circular disk; "
            "and the button terminates the task after the requested repetitions."
        ),
    ),
    operational_notes=(
        (
            "A pick-and-place is successful when the robot lifts the cube above the "
            "predefined height threshold while maintaining a valid grasp, then "
            "lowers it onto the target surface below the specified placement height."
        ),
    ),
)


SWING_XTIMES_DESCRIPTION = TaskDescription(
    task_name="SwingXtimes",
    summary=(
        "The SwingXtimes task. The robot must pick up a specified colored cube, "
        "transport it back and forth between two distinct targets (right-side and "
        "left-side) for a specified number of repetitions, and then press a button "
        "to stop."
    ),
    task_goal_examples=(
        (
            "Pick up the green cube, move it to the right-side target and then put "
            "down the cube on the left-side target and press the button to stop."
        ),
        (
            "Pick up the green cube, move it to the right-side target and then to "
            "the left-side target, repeating this right-to-left swing motion three "
            "times, then put down the cube and press the button to stop."
        ),
    ),
    object_roles=(
        (
            "Object-role identities: the manipulated object is the requested colored "
            "cube; the destinations are two circular target disks; and the button "
            "terminates the task after the requested swing cycles."
        ),
    ),
    operational_notes=(
        (
            "The target currently beneath the cube may turn red while the other "
            "target remains gray; when the cube moves above the other target, that "
            "target may turn red and the previous target may return to gray."
        ),
    ),
)


BUTTON_UNMASK_DESCRIPTION = TaskDescription(
    task_name="ButtonUnmask",
    summary=(
        "The ButtonUnmask task evaluates the robot’s spatial memory for identifying "
        "hidden objects through active interaction. Unlike passive observation tasks, "
        "the robot must press a button while memorizing the visible colored-cube "
        "positions before white containers return to cover them, then determine which "
        "container hides the specified cube and pick it up."
    ),
    task_goal_examples=(
        (
            "First press the button on the table, then pick up the container hiding "
            "the green cube."
        ),
        (
            "First press the button on the table, then pick up the container hiding "
            "the green cube, and finally pick up the container hiding the red cube."
        ),
    ),
    object_roles=(
        (
            "Object-role identities: the button initiates the hiding event; the red, "
            "green, and blue cubes are the hidden identity cues; and the white "
            "containers are the objects selected and picked during execution."
        ),
    ),
    operational_notes=(
        (
            "Button pressing: press the button at the beginning. Early during the "
            "press, the colored cubes are visible; the containers then return to the "
            "table and hide them. Some containers may be empty. Memorize which "
            "container position corresponds to each cube before the cubes are hidden."
        ),
        (
            "Unmasking: identify and pick up the container or containers hiding the "
            "specified cubes. When multiple pickups are required, release the first "
            "container onto the table before attempting the next."
        ),
    ),
)


VIDEO_UNMASK_DESCRIPTION = TaskDescription(
    task_name="VideoUnmask",
    summary=(
        "The VideoUnmask task evaluates spatial memory from a demonstration video. "
        "The robot first watches the video and infers which container hides the "
        "specified cube (red, blue, or green), then selects the corresponding "
        "container(s) during execution. Successful completion requires identifying "
        "the correct container(s) from the video and picking each required one at "
        "least once (in the specified order when multiple picks are requested), "
        "without selecting any incorrect containers."
    ),
    task_goal_examples=(
        "Watch the video carefully, then pick up the container hiding the green cube.",
        (
            "Watch the video carefully, then pick up the container hiding the green "
            "cube, and finally pick up the container hiding the red cube."
        ),
    ),
    object_roles=(
        (
            "Object-role identities: the red, green, and blue cubes provide the "
            "hidden identities, while the containers are the objects that cover the "
            "cubes and are later picked during execution."
        ),
    ),
    operational_notes=(
        (
            "Video: multiple containers are placed on the table. Each colored cube "
            "is hidden under a container, and some containers may be empty. The robot "
            "remains stationary during the video."
        ),
        (
            "Execution: pick up the container or containers hiding the specified "
            "cubes, using the correspondence inferred from the demonstration."
        ),
    ),
)


VIDEO_UNMASK_SWAP_DESCRIPTION = TaskDescription(
    task_name="VideoUnmaskSwap",
    summary=(
        "The VideoUnmaskSwap task evaluates spatial memory from a video "
        "demonstration. The robot observes containers swapping positions, tracks the "
        "correct containers, and then retrieves them during execution."
    ),
    task_goal_examples=(
        "Watch the video carefully, then pick up the container hiding the blue cube.",
        (
            "Watch the video carefully, then pick up the container hiding the blue "
            "cube, and finally pick up the container hiding the red cube."
        ),
    ),
    object_roles=(
        (
            "Object-role identities: the red, green, and blue cubes provide the "
            "hidden identities, while the moving containers cover the cubes and are "
            "the objects selected during execution."
        ),
    ),
    operational_notes=(
        (
            "Video: multiple containers are placed on the table. Each colored cube "
            "is hidden under a container, some containers may be empty, and the "
            "containers swap positions multiple times while the robot remains still."
        ),
        (
            "Execution: track each container’s identity across the observed swaps and "
            "pick the containers hiding the specified cubes in the requested order."
        ),
    ),
)


BUTTON_UNMASK_SWAP_DESCRIPTION = TaskDescription(
    task_name="ButtonUnmaskSwap",
    summary=(
        "The ButtonUnmaskSwap task evaluates spatial memory under active interaction. "
        "The robot must press both buttons during container hiding and swapping, "
        "track the container concealing the target cube, and finally lift the "
        "correct one."
    ),
    task_goal_examples=(
        (
            "First press both buttons on the table, then pick up the container "
            "hiding the red cube."
        ),
        (
            "First press both buttons on the table, then pick up the container "
            "hiding the green cube, and finally pick up the container hiding the "
            "red cube."
        ),
    ),
    object_roles=(
        (
            "Object-role identities: two buttons trigger the hiding and swapping "
            "events; the red, green, and blue cubes provide the hidden identities; "
            "and the containers are the objects tracked and picked."
        ),
    ),
    operational_notes=(
        (
            "Button pressing: press both buttons. Early during pressing, the colored "
            "cubes are visible; the containers then cover them and keep them hidden "
            "while some containers swap positions. Remember the initial cube-to-"
            "container correspondence and track the same containers across the swaps."
        ),
        (
            "Unmasking: pick up the containers hiding the specified cube colors, in "
            "the requested order when multiple pickups are required."
        ),
    ),
)


PICK_HIGHLIGHT_DESCRIPTION = TaskDescription(
    task_name="PickHighlight",
    summary=(
        "The PickHighlight task evaluates the robot’s shortterm object memory. The "
        "robot must press a button to briefly highlight specific cubes, memorize "
        "the highlighted cubes, and then pick them up after the cues disappear. "
        "This tests the robot’s ability to maintain a working memory of transient "
        "object states."
    ),
    task_goal_examples=(
        (
            "First press the button, then pick up all cubes that have been "
            "highlighted with white areas on the table."
        ),
    ),
    object_roles=(
        (
            "Object-role identities: the button triggers the transient reveal, and "
            "the target objects are colored cubes identified by the white highlighted "
            "areas that briefly appear on them."
        ),
    ),
    operational_notes=(
        (
            "Button pressing: first press the button. During the press, the specified "
            "cubes are briefly highlighted with a white visual indicator for a fixed "
            "number of timesteps."
        ),
        (
            "Highlighted-cube picking: identify and pick up every cube highlighted "
            "during this window. If multiple cubes are specified, each must be picked "
            "up at least once."
        ),
    ),
)


VIDEO_REPICK_DESCRIPTION = TaskDescription(
    task_name="VideoRepick",
    summary=(
        "The VideoRepick task . The robot must watch a demonstration video to "
        "identify a specific cube instance, locate the same instance in the current "
        "scene (even if the cubes have moved or swapped), and repeat the "
        "pick-and-place action exactly N times before pressing a stop button."
    ),
    task_goal_examples=(
        (
            "Watch the video carefully, then pick up the same cube that was "
            "previously picked twice, and finally press the button to stop."
        ),
    ),
    object_roles=(
        (
            "Object-role identities: one demonstrated cube instance is the tracked "
            "target among the candidate cubes, and the button terminates execution "
            "after the requested repetitions."
        ),
    ),
    operational_notes=(
        (
            "Video: the robot picks up and places a specific cube. The robot then "
            "remains still while the cubes may swap positions, so track the same "
            "cube instance through the video."
        ),
        (
            "Execution: locate the same previously picked cube and repeat the "
            "pick-and-place action for the number of cycles specified in the task "
            "goal."
        ),
    ),
)


VIDEO_PLACE_BUTTON_DESCRIPTION = TaskDescription(
    task_name="VideoPlaceButton",
    summary=(
        "The VideoPlaceButton task tests long-video temporal reasoning and object "
        "grounding. The robot must watch a long video in which a cube is placed on "
        "multiple targets, separated by a distinct button press, and then parse a "
        "natural language instruction specifying a temporal relation (e.g., before "
        "or after the button press) to identify the correct destination."
    ),
    task_goal_examples=(
        (
            "Watch the video carefully and place the blue cube on the target where "
            "it was placed immediately before the button was pressed."
        ),
    ),
    object_roles=(
        (
            "Object-role identities: the manipulated object is the requested colored "
            "cube, the possible destinations are purple target disks, and the button "
            "is a temporal landmark in the demonstration rather than an execution "
            "target."
        ),
    ),
    operational_notes=(
        (
            "Video: the robot picks up a cube and alternates between placing it on "
            "different targets and pressing a button. In hard mode, two targets swap "
            "positions after the sequence finishes."
        ),
        (
            "Execution: identify the correct cube, pick it up, and place it on the "
            "target selected by the requested temporal relation, such as immediately "
            "before or after the button press. Do not press the button during "
            "execution."
        ),
    ),
)


VIDEO_PLACE_ORDER_DESCRIPTION = TaskDescription(
    task_name="VideoPlaceOrder",
    summary=(
        "The VideoPlaceOrder task evaluates long-video temporal reasoning and object "
        "grounding. The robot must watch a long video to memorize the sequence of "
        "cube placements, then parse a natural language instruction specifying an "
        "ordinal position (e.g., “the third target”) to identify the correct "
        "destination."
    ),
    task_goal_examples=(
        (
            "Watch the video carefully and place the red cube on the third target "
            "where it was placed."
        ),
    ),
    object_roles=(
        (
            "Object-role identities: the manipulated object is the requested colored "
            "cube, the possible destinations are purple target disks, and the button "
            "separates placements in the demonstration but is not pressed during "
            "execution."
        ),
    ),
    operational_notes=(
        (
            "Video: the robot picks up a cube and places it onto multiple targets in "
            "a specific sequence, pressing a button between placements. In hard mode, "
            "two targets swap positions after the sequence finishes."
        ),
        (
            "Execution: identify and pick up the correct cube, then place it onto the "
            "target corresponding to the requested ordinal rank, such as the third "
            "target visited."
        ),
    ),
)


MOVE_CUBE_DESCRIPTION = TaskDescription(
    task_name="MoveCube",
    summary=(
        "The MoveCube task assesses the robot’s ability to recognize and reproduce "
        "distinct action semantics. The task requires the robot to observe a video "
        "demonstration where a cube is moved to a target using one of three specific "
        "manipulation manners: (1) hooking it with a tool; (2) pushing it with a "
        "closed gripper; or (3) performing a standard pick-and-place operation. "
        "Success requires correctly inferring the underlying motion strategy from "
        "videos and executing the exact same method during the execution phase."
    ),
    task_goal_examples=(
        (
            "Watch the video carefully, then move the cube to the target in the "
            "same manner as before."
        ),
    ),
    object_roles=(
        (
            "Object-role identities: the target is the purple circular "
            "disk, the cube is the small compact block, and the peg or tool is the "
            "long thin object."
        ),
    ),
    operational_notes=(
        (
            "Video: the robot moves a cube to a target using one of three methods: "
            "(1) Hooking—pick up a peg tool and use it to hook the cube to the target; "
            "(2) Pushing—close the gripper and push the cube across the surface to the "
            "target; (3) Pick-and-place—grasp the cube directly, lift it, and place it "
            "on the target."
        ),
        (
            "Execution: after the demonstration, the cube and target are reset to new "
            "random positions. Identify the method used in the video and replicate it "
            "exactly while adapting it to the new configuration."
        ),
    ),
)


INSERT_PEG_DESCRIPTION = TaskDescription(
    task_name="InsertPeg",
    summary=(
        "The InsertPeg task evaluates fine-grained visual perception and "
        "procedural/object memory. The robot observes a demonstration to determine "
        "which peg to select, which end to grasp, and which side to insert, then "
        "reproduces the same manipulation after the scene is reset to the same "
        "initial configuration as in the video."
    ),
    task_goal_examples=(
        (
            "Watch the video carefully, then grasp the same peg at the same end and "
            "insert it into the same side of the box as in the video."
        ),
    ),
    object_roles=(
        (
            "Object-role identities: the manipulated object is the elongated peg, "
            "whose near and far ends must be distinguished, and the destination is "
            "the box with robot-left and robot-right insertion sides."
        ),
    ),
    operational_notes=(
        (
            "Video: the robot picks up a specific peg by grasping its head or tail "
            "end and inserts it into the box from a specific side. After the "
            "demonstration, all pegs return to their original positions and "
            "orientations."
        ),
        (
            "Execution: identify the demonstrated peg, grasp the same end, and insert "
            "it into the box from the demonstrated side. Interpret left and right in "
            "the robot base frame rather than the front-camera image."
        ),
    ),
)


PATTERN_LOCK_DESCRIPTION = TaskDescription(
    task_name="PatternLock",
    summary=(
        "The PatternLock task evaluates the robot’s longhorizon procedural memory "
        "from a video demonstration. The robot observes a stick attached to the "
        "end-effector tracing a continuous pattern over an NxN grid of gray targets, "
        "and must then reproduce the same trajectory during execution. Success "
        "requires following the full demonstrated sequence without skipping or "
        "moving to incorrect targets."
    ),
    task_goal_examples=(
        (
            "Watch the video carefully, then use the stick attached to the robot to "
            "retrace the same pattern shown in the video."
        ),
    ),
    object_roles=(
        (
            "Object-role identities: the tracing tool is the stick attached to the "
            "end-effector, and the waypoints are the gray or white circular targets "
            "arranged in an NxN grid."
        ),
    ),
    operational_notes=(
        "Demonstration: reconstruct the complete ordered target sequence.",
        (
            "Execution: retrace the targets continuously in the same order. A target "
            "turns red when the stick reaches it, and a line renders the stick-tip "
            "trajectory in real time."
        ),
        (
            "Use the robot base frame for directions. In the front camera, robot "
            "forward appears backward, robot backward appears forward, robot left "
            "appears right, and robot right appears left."
        ),
    ),
)


ROUTE_STICK_DESCRIPTION = TaskDescription(
    task_name="RouteStick",
    summary=(
        "The RouteStick task evaluates long-horizon, geometry-aware procedural "
        "memory from a video demonstration. The robot observes a demonstration in "
        "which a stick attached to the endeffector navigates through a field of "
        "fixed vertical obstacles while visiting a sequence of targets. Successful "
        "completion requires circling each obstacle in the demonstrated direction "
        "(clockwise or counterclockwise) and following a single continuous trajectory."
    ),
    task_goal_examples=(
        (
            "Watch the video carefully, then use the stick attached to the robot to "
            "navigate around the obstacles on the table, following the same path "
            "shown in the video."
        ),
    ),
    object_roles=(
        (
            "Object-role identities: the moving tool is the stick attached to the "
            "end-effector, the route constraints are the fixed vertical obstacles, "
            "and the destinations are the target markers visited in sequence."
        ),
    ),
    operational_notes=(
        (
            "Video: infer the ordered targets and the clockwise or counterclockwise "
            "direction used to circle each obstacle."
        ),
        (
            "Execution: visit the same targets in the same order, reproduce the "
            "demonstrated circling directions continuously, use target color changes "
            "as visit feedback, and use the rendered stick-tip trajectory as path "
            "feedback."
        ),
        (
            "Use the robot base frame for directions. In the front camera, robot "
            "forward appears backward, robot backward appears forward, robot left "
            "appears right, and robot right appears left."
        ),
    ),
)


_TASK_DESCRIPTIONS: tuple[TaskDescription, ...] = (
    BINFILL_DESCRIPTION,
    STOP_CUBE_DESCRIPTION,
    PICK_XTIMES_DESCRIPTION,
    SWING_XTIMES_DESCRIPTION,
    BUTTON_UNMASK_DESCRIPTION,
    VIDEO_UNMASK_DESCRIPTION,
    VIDEO_UNMASK_SWAP_DESCRIPTION,
    BUTTON_UNMASK_SWAP_DESCRIPTION,
    PICK_HIGHLIGHT_DESCRIPTION,
    VIDEO_REPICK_DESCRIPTION,
    VIDEO_PLACE_BUTTON_DESCRIPTION,
    VIDEO_PLACE_ORDER_DESCRIPTION,
    MOVE_CUBE_DESCRIPTION,
    INSERT_PEG_DESCRIPTION,
    PATTERN_LOCK_DESCRIPTION,
    ROUTE_STICK_DESCRIPTION,
)

_TASK_DESCRIPTION_BY_KEY = {
    description.task_name.casefold(): description
    for description in _TASK_DESCRIPTIONS
}


def available_task_descriptions() -> tuple[str, ...]:
    """Return registered task names in deterministic order."""

    return tuple(description.task_name for description in _TASK_DESCRIPTIONS)


def get_task_description(task_name: str) -> TaskDescription:
    """Resolve one description with case-insensitive task-name matching."""

    key = task_name.strip().casefold()
    if not key:
        raise ValueError("task_name must be non-empty")
    try:
        return _TASK_DESCRIPTION_BY_KEY[key]
    except KeyError as error:
        available = ", ".join(available_task_descriptions())
        raise KeyError(
            f"No task description registered for {task_name!r}; "
            f"available: {available}"
        ) from error
