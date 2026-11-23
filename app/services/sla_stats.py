"""
Empirical SLA-breach estimation from real historical resolution durations.

Pure Python (no DB / web imports) so it is unit-testable and auditable.

Model ("survival-style", non-parametric):
  For an open case that is already `elapsed` days old with deadline `sla` days,
      P(breach) = P(total duration > sla | total duration > elapsed)
  estimated from comparable historical cases (same category -> else same department -> else all),
  with Laplace smoothing. If the deadline has already passed the case is reported as breached.

Duration = resolution_date - submission_date of rows whose status is Resolved / Closed
(compared case-insensitively; the mock CSV uses "Resolved" / "Closed").
"""
from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Dict, Iterable, List, Optional, Tuple

RESOLVED_STATUSES = {"resolved", "closed", "citizen_satisfied"}


def parse_date(value) -> Optional[datetime]:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value.replace(tzinfo=None)
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day)
    text = str(value).strip().replace("T", " ")
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(text[:26] if "%f" in fmt else text[:19] if "%S" in fmt else text[:10], fmt)
        except ValueError:
            continue
    return None


def duration_days(submission, resolution) -> Optional[float]:
    s, r = parse_date(submission), parse_date(resolution)
    if s is None or r is None:
        return None
    d = (r - s).total_seconds() / 86400.0
    return d if d >= 0 else None


def _percentile(sorted_vals: List[float], q: float) -> float:
    if not sorted_vals:
        return float("nan")
    idx = (len(sorted_vals) - 1) * q
    lo, hi = int(idx), min(int(idx) + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (idx - lo)


@dataclass
class DurationTable:
    by_category: Dict[str, List[float]] = field(default_factory=dict)
    by_department: Dict[str, List[float]] = field(default_factory=dict)
    overall: List[float] = field(default_factory=list)

    @property
    def size(self) -> int:
        return len(self.overall)


def build_duration_table(rows: Iterable[Tuple[str, str, str, object, object]]) -> DurationTable:
    """rows: (category_id, department_id, status, submission_date, resolution_date)."""
    t = DurationTable()
    for category_id, department_id, status, sub, res in rows:
        if (status or "").strip().lower() not in RESOLVED_STATUSES:
            continue
        d = duration_days(sub, res)
        if d is None:
            continue
        t.by_category.setdefault(category_id, []).append(d)
        t.by_department.setdefault(department_id, []).append(d)
        t.overall.append(d)
    for lst in (*t.by_category.values(), *t.by_department.values(), t.overall):
        lst.sort()
    return t


@dataclass
class BreachEstimate:
    probability: float
    method: str
    pool: str
    n_samples: int
    n_conditional: int
    median_days: Optional[float]
    p90_days: Optional[float]
    historical_breach_rate: Optional[float]
    already_overdue: bool
    backlog_adjustment: float = 0.0
    recurrence_adjustment: float = 0.0

    def explain(self, sla_days: float, elapsed_days: float, label: str) -> str:
        if self.already_overdue:
            return (
                f"SLA breached: case is {elapsed_days:.1f} days old against a {sla_days:g}-day SLA "
                f"and is still open."
            )
        if self.method == "no_history":
            base = "No comparable resolved cases in the history; probability is an uninformed prior."
        else:
            rate = f"{self.historical_breach_rate * 100:.0f}%" if self.historical_breach_rate is not None else "n/a"
            cond = (
                f"Of {self.n_conditional} historical {label} cases still open at {elapsed_days:.1f} days, "
                f"{self.probability * 100:.0f}% went on to exceed the {sla_days:g}-day SLA"
                if self.method.startswith("conditional")
                else f"{rate} of {self.n_samples} historical {label} cases exceeded the {sla_days:g}-day SLA"
            )
            med = f" (median resolution {self.median_days:.1f} days, p90 {self.p90_days:.1f})" if self.median_days is not None else ""
            base = f"{cond}{med}."
        extras = []
        if self.backlog_adjustment:
            extras.append(f"department backlog +{self.backlog_adjustment:.2f}")
        if self.recurrence_adjustment:
            extras.append(f"recurring same-area reports +{self.recurrence_adjustment:.2f}")
        tail = f" Adjustments: {', '.join(extras)}." if extras else ""
        return f"Pre-emptive SLA warning: {base}{tail} Demo SLA values from the category master."


def _pool_stats(durations: List[float], sla_days: float) -> Tuple[float, float, float]:
    n = len(durations)
    breach_rate = sum(1 for d in durations if d > sla_days) / n
    return _percentile(durations, 0.5), _percentile(durations, 0.9), breach_rate


def estimate_breach(
    table: DurationTable,
    category_id: Optional[str],
    department_id: Optional[str],
    sla_days: float,
    elapsed_days: float,
    backlog_open: int = 0,
    recurring: bool = False,
    min_samples: int = 5,
) -> BreachEstimate:
    if elapsed_days >= sla_days:
        return BreachEstimate(1.0, "already_overdue", "none", 0, 0, None, None, None, True)

    pools = [
        ("category", table.by_category.get(category_id or "", [])),
        ("department", table.by_department.get(department_id or "", [])),
        ("overall", table.overall),
    ]
    est: Optional[BreachEstimate] = None
    for pool_name, durations in pools:
        if len(durations) < min_samples:
            continue
        median, p90, breach_rate = _pool_stats(durations, sla_days)
        survivors = [d for d in durations if d > elapsed_days]
        if len(survivors) >= min_samples:
            exceed = sum(1 for d in survivors if d > sla_days)
            p = (exceed + 1) / (len(survivors) + 2)                    # Laplace smoothing
            method = f"conditional_{pool_name}"
            n_cond = len(survivors)
        else:
            # Case has outlived (almost) all comparable history: fall back to the unconditional rate
            exceed = sum(1 for d in durations if d > sla_days)
            p = (exceed + 1) / (len(durations) + 2)
            method = f"unconditional_{pool_name}"
            n_cond = len(survivors)
        est = BreachEstimate(p, method, pool_name, len(durations), n_cond, median, p90, breach_rate, False)
        break

    if est is None:
        est = BreachEstimate(0.5, "no_history", "none", 0, 0, None, None, None, False)

    backlog_adj = round(min(max(backlog_open, 0) * 0.01, 0.10), 3)
    recur_adj = 0.05 if recurring else 0.0
    est.backlog_adjustment = backlog_adj
    est.recurrence_adjustment = recur_adj
    est.probability = round(min(est.probability + backlog_adj + recur_adj, 0.99), 3)
    return est
