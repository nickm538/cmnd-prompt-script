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
    return str(value)


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
            lines.append(f"- {headline}")

    lines.append("")

    rows = report.get("top_25") or []
    if not rows:
        lines.append("No candidates passed the pipeline.")
        return lines

    lines.append("| # | Ticker | Conf | ML | Panel | Hype | Exh | Insider | Action |")
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
        "_Conf = overall confidence, Exh = exhaustion (lower is better; "
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
