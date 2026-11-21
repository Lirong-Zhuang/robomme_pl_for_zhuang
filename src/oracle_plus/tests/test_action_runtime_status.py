from oracle_plus.action_runtime import resolve_robomme_status


def test_reported_terminal_status_is_authoritative() -> None:
    assert (
        resolve_robomme_status(
            reported_status="error",
            evaluation={"success": True, "fail": False},
            terminated=True,
            truncated=False,
        )
        == "error"
    )
    assert (
        resolve_robomme_status(
            reported_status="timeout",
            evaluation={"success": False, "fail": True},
            terminated=False,
            truncated=True,
        )
        == "timeout"
    )


def test_fallback_matches_robomme_flag_order() -> None:
    assert (
        resolve_robomme_status(
            reported_status="ongoing",
            evaluation={"success": True, "fail": False},
            terminated=True,
            truncated=True,
        )
        == "success"
    )
    assert (
        resolve_robomme_status(
            reported_status=None,
            evaluation={"success": False, "fail": False},
            terminated=True,
            truncated=True,
        )
        == "fail"
    )
    assert (
        resolve_robomme_status(
            reported_status=None,
            evaluation={"success": False, "fail": False},
            terminated=False,
            truncated=True,
        )
        == "timeout"
    )
    assert (
        resolve_robomme_status(
            reported_status="ongoing",
            evaluation={"success": False, "fail": False},
            terminated=False,
            truncated=False,
        )
        == "ongoing"
    )
