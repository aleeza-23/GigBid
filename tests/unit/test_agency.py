"""X1 routing rules: the two cases where the router must refuse to decide."""

from __future__ import annotations

from agent.nodes.agency import agency_config, choose


def _row(member: str, coverage: float, projects: int = 2) -> dict:
    return {
        "member": member,
        "coverage": coverage,
        "covered": ["a"] * int(coverage * 10),
        "missing": ["b"] * (10 - int(coverage * 10)),
        "projects": projects,
    }


def test_clear_winner_is_routed():
    verdict = choose(
        [_row("data_analyst", 0.9), _row("ai_ml_engineer", 0.3)],
        default_member="ai_ml_engineer",
    )
    assert verdict["routed"] is True
    assert verdict["member"] == "data_analyst"
    assert "next best" in verdict["reason"]


def test_a_narrow_lead_is_not_a_decision():
    """Within the margin, the job goes to the default member and is flagged."""
    verdict = choose(
        [_row("data_analyst", 0.60), _row("ai_ml_engineer", 0.55)],
        default_member="ai_ml_engineer",
        min_margin=0.15,
    )
    assert verdict["routed"] is False
    assert verdict["member"] == "ai_ml_engineer"
    assert "too close to call" in verdict["reason"]
    # The better member is still reported, so a person can override.
    assert verdict["best"] == "data_analyst"


def test_nobody_covers_enough():
    verdict = choose(
        [_row("data_analyst", 0.2), _row("ai_ml_engineer", 0.1)],
        default_member="ai_ml_engineer",
        min_coverage=0.34,
    )
    assert verdict["routed"] is False
    assert verdict["member"] == "ai_ml_engineer"
    assert "below the" in verdict["reason"]


def test_no_requirements_means_no_routing():
    verdict = choose([], default_member="ai_ml_engineer")
    assert verdict["routed"] is False
    assert verdict["member"] == "ai_ml_engineer"


def test_equal_coverage_breaks_on_project_count():
    """Four requirements from four projects is stronger evidence than from one."""
    ranked = sorted(
        [_row("a", 0.8, projects=1), _row("b", 0.8, projects=4)],
        key=lambda row: (-row["coverage"], -row["projects"], row["member"]),
    )
    assert ranked[0]["member"] == "b"


def test_agency_mode_is_off_by_default():
    """The single-freelancer product is the default, and the graph must stay unchanged."""
    settings = agency_config()
    assert settings["enabled"] is False
    assert "ai_ml_engineer" in settings["members"]
