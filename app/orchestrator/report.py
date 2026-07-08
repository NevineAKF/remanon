"""
Band A, Layer L8 — incident report builder.

The agents' final product is not a JSON payload on a screen; it is a file
an on-call engineer can hand to their team. This module turns the same
EventLog the L9 dashboard already reads into a structured incident report,
and renders it as CSV (one row per case, opens in Excel) and Markdown (a
professional write-up with a title block, run summary, and one section per
case). Read-only: it only ever reads an EventLog and a metrics snapshot.
"""

from __future__ import annotations

import csv
import io
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from app.orchestrator.orchestrator import Event

_ROOT_CAUSE_RE = re.compile(r"root cause:?\s*(.+)", re.IGNORECASE | re.DOTALL)
_CAUSE_HEADING_RE = re.compile(r"finding|cause", re.IGNORECASE)
_RECOMMEND_HEADING_RE = re.compile(r"recommend|mitigat|action", re.IGNORECASE)

# Agents in fixed display order — mirrors the dashboard's five-dot row.
AGENT_ORDER = ("triage", "correlator", "hunter", "topology", "reporter")

CSV_FIELDS = [
    "case_id",
    "opened_at",
    "severity",
    "title",
    "trigger_count",
    "nodes",
    "category",
    "root_cause",
    "evidence",
    "blast_radius_nodes",
    "blast_radius_edges",
    "blast_radius_labels",
    "recommendation",
    "contributing_agents",
    "degraded_agents",
    "working_delta_gb",
    "masters_gb_reused",
]


# ---------------------------------------------------------------------------
# extraction — the same heuristics the live theater ledger uses, in Python
# ---------------------------------------------------------------------------


def _root_cause_of(payload: dict[str, Any]) -> str:
    summary = payload.get("executive_summary") or ""
    match = _ROOT_CAUSE_RE.search(summary)
    if match:
        return match.group(1).strip()
    for section in payload.get("sections") or []:
        if _CAUSE_HEADING_RE.search(section.get("heading") or ""):
            return section.get("body") or "—"
    return summary or "—"


def _recommendation_of(payload: dict[str, Any]) -> str:
    for section in payload.get("sections") or []:
        if _RECOMMEND_HEADING_RE.search(section.get("heading") or ""):
            return section.get("body") or "—"
    return "—"


def extract_cases_from_events(
    events: list[Event], ledger: dict[str, Any] | None = None
) -> list[dict[str, Any]]:
    """
    Rebuild one report row per closed case from a raw EventLog stream.

    Only cases that reached a reporter verdict are included — an in-flight
    or degraded-to-nothing case has no root cause to report yet.
    """
    ledger = ledger or {}
    by_case: dict[str, dict[str, Any]] = {}
    order: list[str] = []

    for event in events:
        case_id = event.case_id
        if case_id not in by_case:
            by_case[case_id] = {
                "triggers": [],
                "opened_at": None,
                "category": None,
                "hunter": None,
                "topo": None,
                "contrib": [],
                "degraded": [],
                "reporter_payload": None,
            }
            order.append(case_id)
        cs = by_case[case_id]
        data = event.data

        if event.kind == "case_open":
            cs["triggers"] = data.get("trigger_records") or []
            cs["opened_at"] = data.get("opened_at")
        elif event.kind == "artifact":
            agent = data.get("agent")
            if agent and agent not in cs["contrib"]:
                cs["contrib"].append(agent)
            try:
                payload = data["artifact_raw"]["payload"]
            except (KeyError, TypeError):
                payload = None
            if payload is None:
                continue
            if agent == "triage":
                cs["category"] = payload.get("category")
            elif agent == "hunter":
                findings = payload.get("findings") or []
                cs["hunter"] = findings[0] if findings else None
            elif agent == "topology":
                cs["topo"] = payload
            elif agent == "reporter":
                cs["reporter_payload"] = payload
        elif event.kind == "degraded":
            agent = data.get("agent")
            if agent and agent not in cs["degraded"]:
                cs["degraded"].append(agent)

    rows: list[dict[str, Any]] = []
    for case_id in order:
        cs = by_case[case_id]
        payload = cs["reporter_payload"]
        if payload is None:
            continue  # no verdict landed — nothing to report yet

        triggers = cs["triggers"]
        nodes = sorted({t.get("node") for t in triggers if t.get("node")})
        hunter = cs["hunter"] or {}
        evidence_lines = hunter.get("evidence") or []
        evidence = evidence_lines[0] if evidence_lines else "—"
        if hunter.get("confidence") is not None:
            evidence += f" · conf {round(hunter['confidence'] * 100)}%"
        topo = cs["topo"] or {}
        topo_nodes = topo.get("nodes") or []
        topo_edges = topo.get("edges") or []

        rows.append(
            {
                "case_id": case_id,
                "opened_at": cs["opened_at"],
                "severity": (payload.get("overall_severity") or "info").lower(),
                "title": payload.get("title") or "",
                "trigger_count": len(triggers),
                "nodes": nodes,
                "category": cs["category"],
                "root_cause": _root_cause_of(payload),
                "evidence": evidence if hunter else "—",
                "blast_radius_nodes": len(topo_nodes),
                "blast_radius_edges": len(topo_edges),
                "blast_radius_labels": [
                    n.get("label") or n.get("node_id") or "" for n in topo_nodes
                ],
                "recommendation": _recommendation_of(payload),
                "contributing_agents": [a for a in AGENT_ORDER if a in cs["contrib"]],
                "degraded_agents": [a for a in AGENT_ORDER if a in cs["degraded"]],
                "working_delta_gb": float(ledger.get("deltas_gb", 0.0)),
                "masters_gb_reused": float(ledger.get("masters_gb", 0.0)),
            }
        )
    return rows


# ---------------------------------------------------------------------------
# report assembly
# ---------------------------------------------------------------------------


def build_incident_report(
    cases: list[dict[str, Any]],
    metrics: dict[str, Any],
    *,
    engine_mode: str = "mock",
    generated_at: datetime | None = None,
) -> dict[str, Any]:
    """Assemble the full incident report: run summary + one row per case."""
    summary = {
        "cases_processed": len(cases),
        "prefills_avoided": metrics.get("prefills_avoided", 0),
        "prefills_performed": metrics.get("prefills_performed", 0),
        "gb_saved_vs_per_agent": metrics.get("gb_saved_vs_per_agent", 0.0),
        "active_leases": metrics.get("active_leases", 0),
        "evictions": 0,  # pinned masters are non-evictable by construction
        "engine_mode": engine_mode,
    }
    return {
        "generated_at": (generated_at or datetime.now(UTC)).isoformat(),
        "engine_mode": engine_mode,
        "summary": summary,
        "cases": cases,
    }


# ---------------------------------------------------------------------------
# rendering — in-memory strings, so API endpoints need no filesystem I/O
# ---------------------------------------------------------------------------


def render_csv_report(report: dict[str, Any]) -> str:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=CSV_FIELDS, extrasaction="ignore")
    writer.writeheader()
    for case in report["cases"]:
        row = dict(case)
        for key in ("nodes", "blast_radius_labels", "contributing_agents", "degraded_agents"):
            row[key] = "; ".join(row.get(key) or [])
        writer.writerow(row)
    return buf.getvalue()


def render_markdown_report(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# REMANON",
        "GPU-resident shared-memory runtime for multi-agent inference",
        "",
        f"**Incident report generated:** {report['generated_at']}",
        f"**Engine mode:** {report['engine_mode']}",
        "",
        "## Run summary",
        "",
        "| Cases closed | Prefills avoided | GB saved | Evictions |",
        "|---|---|---|---|",
        f"| {summary['cases_processed']} | {summary['prefills_avoided']} "
        f"| {summary['gb_saved_vs_per_agent']:.1f} | {summary['evictions']} |",
        "",
    ]

    for case in report["cases"]:
        lines += [
            f"## Case {case['case_id'][:8]} — {case['severity'].upper()}",
            "",
            f"**Title:** {case['title']}" if case.get("title") else None,
            f"- **Opened:** {case['opened_at'] or '—'}",
            f"- **Problem:** {case['trigger_count']} alerts in window"
            + (f" · {', '.join(case['nodes'])}" if case["nodes"] else "")
            + (f" · {case['category']}" if case.get("category") else ""),
            f"- **Root cause:** {case['root_cause']}",
            f"- **Evidence:** {case['evidence']}",
            f"- **Blast radius:** {case['blast_radius_nodes']} node(s) · "
            f"{case['blast_radius_edges']} link(s)"
            + (
                f" — {', '.join(case['blast_radius_labels'])}"
                if case["blast_radius_labels"]
                else ""
            ),
            f"- **Recommendation:** {case['recommendation']}",
            f"- **Contributing agents:** {', '.join(case['contributing_agents']) or '—'}",
            f"- **Degraded agents:** {', '.join(case['degraded_agents']) or 'none'}",
            f"- **Memory cost:** Δ{case['working_delta_gb']:.1f} GB working "
            f"· {case['masters_gb_reused']:.0f} GB reused",
            "",
        ]
        lines = [line for line in lines if line is not None]

    return "\n".join(lines) + "\n"


def export_incident_report(
    report: dict[str, Any], out_dir: Path, *, timestamp: str | None = None
) -> tuple[Path, Path]:
    """Write both formats to *out_dir*; returns (csv_path, md_path)."""
    ts = timestamp or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"incident_report_{ts}.csv"
    md_path = out_dir / f"incident_report_{ts}.md"
    csv_path.write_text(render_csv_report(report), encoding="utf-8", newline="")
    md_path.write_text(render_markdown_report(report), encoding="utf-8")
    return csv_path, md_path
