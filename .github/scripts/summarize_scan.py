#!/usr/bin/env python3
"""Render a scan report as a GitHub Actions job summary (Markdown).

Usage: summarize_scan.py <path-to-scan_*.json>

Kept deliberately dependency-free so it can run before/without the scanner's
own requirements being importable, and so a malformed report degrades into a
readable note instead of failing the workflow.
"""

from __future__ import annotations

import json
import sys
from typing import Any, Dict, List


def _cell(value: Any) -> str:
    """Render a table cell, tolerating missing/None values."""
    if value is None or value == "":
        return "--"
    return str(value).replace("|", "\\|").replace("\r", " ").replace("\n", " ")


def _text(value: Any) -> str:
    """Keep untrusted provider text inside one Markdown line."""
    return str(value or "").replace("\r", " ").replace("\n", " ").strip()


def _render(report: Dict[str, Any]) -> List[str]:
    strategy = report.get("strategy") or {}
    lines: List[str] = [
        "",
        f"### Scan {report.get('scan_timestamp', 'unknown time')}",
        "",
        f"- Survivors: {report.get('total_survivors', 0)}",
    ]

    horizon = strategy.get("holding_horizon_days")
    if horizon is not None:
        lines.append(f"- Holding horizon: {horizon} sessions")

    dte_window = strategy.get("option_dte_window")
    if isinstance(dte_window, list) and len(dte_window) == 2:
        lines.append(f"- Option DTE window: {dte_window[0]}-{dte_window[1]} days")

    macro = report.get("macro_regime") or {}
    if macro.get("regime_label"):
        lines.append(
            f"- Macro regime: {macro['regime_label']} "
            f"({macro.get('regime_score', '?')}/100)"
        )

    world = macro.get("world_context") or {}
    if world.get("event_risk") is not None:
        lines.append(f"- Live event risk: {world.get('event_risk')}/100")
    headlines = world.get("headlines") or []
    if headlines:
        lines.append("")
        lines.append("Today's market headlines (context only; not a ticker list):")
        for item in headlines[:5]:
            headline = item.get("headline") if isinstance(item, dict) else item
            lines.append(f"- {_text(headline)}")
    economic_events = [
        event for event in (world.get("economic_events") or [])
        if isinstance(event, dict) and event.get("high_impact")
    ]
    if economic_events:
        lines.append("")
        lines.append("High-impact US macro calendar:")
        for event in economic_events[:5]:
            lines.append(
                f"- {_text(event.get('time'))}: "
                f"{_text(event.get('event'))}"
            )
    if world.get("context_degraded"):
        statuses = world.get("feed_status") or {}
        degraded = ", ".join(
            f"{name}={status}"
            for name, status in statuses.items()
            if status != "ok"
        )
        lines.append(f"- Context coverage degraded: {_text(degraded)}")

    lines.append("")

    rows = report.get("top_25") or []
    near_misses = report.get("near_misses") or []
    if report.get("report_kind") == "incomplete":
        lines.append(
            "Scanner run incomplete: "
            + _text(report.get("reason") or "unknown failure")
        )
        return lines
    if report.get("report_kind") == "no_action" or (
        not rows and near_misses
    ):
        lines.append(
            "No actionable candidate passed every hard rule. "
            "This is a valid abstention, not a failed scan."
        )
        if near_misses:
            lines.append("")
            lines.append("Closest non-actionable setups:")
            lines.append("")
            lines.append("| # | Ticker | Rules | Failed |")
            lines.append("|---|--------|-------|--------|")
            for row in near_misses[:7]:
                failed = row.get("failed") or []
                if isinstance(failed, list):
                    failed = ", ".join(str(item) for item in failed)
                lines.append(
                    "| {rank} | {ticker} | {rules}/10 | {failed} |".format(
                        rank=_cell(row.get("rank")),
                        ticker=_cell(row.get("ticker")),
                        rules=_cell(row.get("rules_passed")),
                        failed=_cell(failed),
                    )
                )
        return lines

    if not rows:
        lines.append("No candidates passed the pipeline.")
        return lines

    lines.append("| # | Ticker | Setup | ML | Panel | Hype | Exh | Insider | Action |")
    lines.append("|---|--------|------|----|-------|------|-----|---------|--------|")
    for row in rows[:7]:
        lines.append(
            "| {rank} | {ticker} | {conf} | {ml} | {panel} | {hype} | {exh} "
            "| {insider} | {action} |".format(
                rank=_cell(row.get("rank")),
                ticker=_cell(row.get("ticker")),
                conf=_cell(row.get("overall_confidence")),
                ml=_cell(row.get("ml_ensemble")),
                panel=_cell(row.get("panel_composite")),
                hype=_cell(row.get("hype_score")),
                exh=_cell(row.get("exhaustion_score")),
                insider=_cell(row.get("insider_score")),
                action=_cell(row.get("action")),
            )
        )

    lines.append("")
    lines.append(
        "_Setup is a heuristic quality score, not a win probability. "
        "Exh = exhaustion (lower is better; "
        "high readings mean the move is already extended)._"
    )
    return lines


def main(argv: List[str]) -> int:
    if len(argv) != 2:
        print("No scan report path supplied.")
        return 0

    try:
        with open(argv[1], encoding="utf-8") as handle:
            report = json.load(handle)
    except (OSError, ValueError) as exc:
        print(f"Could not read scan report `{argv[1]}`: {exc}")
        return 0

    if not isinstance(report, dict):
        print(f"Scan report `{argv[1]}` is not a JSON object.")
        return 0

    print("\n".join(_render(report)))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
