"""Global operational knowledge supplied to init planning by default."""

from __future__ import annotations


GRASP_CLEARANCE_NOTE = (
    "For any grasp-related task, securely grasp and lift the small object to a "
    "visibly safe clearance above its supporting surface before starting any "
    "non-vertical motion. Begin horizontal or lateral movement only after the "
    "lift is confirmed, to avoid collisions with the table or nearby objects."
)


DEFAULT_GENERAL_NOTES: tuple[str, ...] = (GRASP_CLEARANCE_NOTE,)
