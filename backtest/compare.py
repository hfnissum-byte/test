from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def rank_backtest_policies(report_base_dir: str | Path = "backtest/results") -> list[tuple[str, dict[str, Any]]]:
    """
    Load `metrics.json` from each policy folder under `report_base_dir` and print a
    ranked table by Sharpe (primary) and net_return (secondary).
    """
    base = Path(report_base_dir)
    if not base.exists():
        print(f"No backtest reports found at: {base}")
        return []

    rows: list[tuple[str, dict[str, Any]]] = []
    for policy_dir in sorted([p for p in base.iterdir() if p.is_dir()]):
        metrics_path = policy_dir / "metrics.json"
        if not metrics_path.exists():
            continue
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        policy_name = metrics.get("policy", policy_dir.name)
        rows.append((str(policy_name), metrics))

    # Rank by sharpe then net return.
    def _key(item: tuple[str, dict[str, Any]]):
        _, m = item
        return (float(m.get("sharpe", 0.0)), float(m.get("net_return", 0.0)))

    ranked = sorted(rows, key=_key, reverse=True)

    # Print a simple text table.
    if ranked:
        print("Policy ranking (Sharpe desc, net_return desc):")
        print(f"{'policy':<18} {'sharpe':>10} {'net_return':>12} {'max_drawdown':>14} {'trades':>10}")
        for name, m in ranked:
            print(
                f"{name:<18} {float(m.get('sharpe', 0.0)):>10.4f} "
                f"{float(m.get('net_return', 0.0)):>12.6f} {float(m.get('max_drawdown', 0.0)):>14.6f} "
                f"{int(float(m.get('trade_count', 0.0))):>10d}"
            )

    return ranked

