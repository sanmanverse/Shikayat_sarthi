"""SLA prediction uses REAL historical durations (regression for the 'RESOLVED' vs 'Resolved' bug)."""
import csv
from pathlib import Path

from app.services.sla_stats import (
    build_duration_table, estimate_breach, duration_days, parse_date, DurationTable,
)

HISTORY = Path(__file__).resolve().parent.parent / "data" / "mock_grievance_history.csv"


def _rows():
    with open(HISTORY, encoding="utf-8-sig") as f:
        return [(r["category_id"], r["department_id"], r["status"], r["submission_date"], r["resolution_date"])
                for r in csv.DictReader(f)]


def test_status_matching_is_case_insensitive_and_uses_real_rows():
    table = build_duration_table(_rows())
    assert table.size > 400, "Resolved/Closed rows must be used (old code matched none)"
    assert all(d >= 0 for d in table.overall)


def test_durations_are_real_not_the_sla_column():
    table = build_duration_table(_rows())
    sla_by_cat = {r[0]: None for r in _rows()}
    c001 = table.by_category["C001"]
    assert len(set(c001)) > 1, "durations must vary; a constant means the SLA column is being averaged"


def test_already_overdue_is_certain():
    e = estimate_breach(build_duration_table(_rows()), "C005", "D002", sla_days=2, elapsed_days=6)
    assert e.already_overdue and e.probability == 1.0


def test_conditional_survival_increases_with_age():
    t = build_duration_table(_rows())
    young = estimate_breach(t, "C001", "D001", 5, 0.0)
    older = estimate_breach(t, "C001", "D001", 5, 2.0)
    assert young.method.startswith("conditional") and older.probability >= young.probability


def test_explanation_cites_the_numbers_actually_used():
    t = build_duration_table(_rows())
    e = estimate_breach(t, "C001", "D001", 5, 1.0, backlog_open=3, recurring=True)
    text = e.explain(5, 1.0, "Potholes")
    assert str(e.n_conditional) in text and "backlog" in text and "recurring" in text


def test_unknown_category_falls_back_to_pool_then_prior():
    t = build_duration_table(_rows())
    assert estimate_breach(t, "CXXX", "DXXX", 7, 1.0).method.startswith(("conditional_overall", "unconditional_overall"))
    assert estimate_breach(DurationTable(), "C001", "D001", 5, 1.0).method == "no_history"


def test_parsers():
    assert parse_date("2025-03-04").day == 4
    assert parse_date("2025-03-04 10:11:12.123456").hour == 10
    assert parse_date(None) is None and duration_days("2025-03-05", "2025-03-04") is None
