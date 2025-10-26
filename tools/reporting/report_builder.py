from __future__ import annotations

import base64
import io
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd


@dataclass
class MetricsSummary:
    compitum_perf: float
    compitum_cost: float
    llm_perf: Dict[str, float]
    llm_cost: Dict[str, float]
    notes: List[str]
    # Regret metrics at selected WTP
    mean_regret: Optional[float] = None
    p95_regret: Optional[float] = None
    win_rate: Optional[float] = None  # fraction of evals where compitum >= best LLM utility
    avg_cost_delta_on_wins: Optional[float] = None  # compitum minus best LLM cost on winning evals
    regrets_by_model: Optional[Dict[str, float]] = None  # mean regret per model at selected/best WTP
    # Moneyshot extras (kept minimal for clarity)


def _fig_to_data_url(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=160)
    plt.close(fig)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def build_metrics_summary(
    compitum_csv: Path,
    wtp: float = 1.0,
    llm_models: Optional[List[str]] = None,
    wtp_list: Optional[List[float]] = None,
) -> MetricsSummary:
    # Auto-discover baselines if not provided: pick up to 8 most represented (or highest perf) routers
    if llm_models is None:
        try:
            candidates = (
                df[df["model_name"] != "compitum"]
                .groupby("model_name")["performance"]
                .mean()
                .sort_values(ascending=False)
            )
            llm_models = [n for n in candidates.index if n.lower() != "oracle"][:8]
        except Exception:
            llm_models = [
                "gpt-3.5-turbo-1106",
                "gpt-4-1106-preview",
                "claude-instant-v1",
                "claude-v1",
                "claude-v2",
            ]
    df = pd.read_csv(compitum_csv)

    # Filter compitum by chosen willingness_to_pay (if present)
    cdf = df[df["model_name"] == "compitum"].copy()
    if "willingness_to_pay" in cdf.columns:
        cdf = cdf[cdf["willingness_to_pay"] == wtp]

    llms = df[df["model_name"].isin(llm_models)].copy()

    metrics = MetricsSummary(
        compitum_perf=float(cdf["performance"].mean()) if not cdf.empty else float("nan"),
        compitum_cost=float(cdf["total_cost"].mean()) if not cdf.empty else float("nan"),
        llm_perf=llms.groupby("model_name")["performance"].mean().to_dict() if not llms.empty else {},
        llm_cost=llms.groupby("model_name")["total_cost"].mean().to_dict() if not llms.empty else {},
        notes=[],
    )
    if cdf.empty:
        metrics.notes.append("No compitum rows found at the selected willingness_to_pay.")
    if llms.empty:
        metrics.notes.append("No baseline LLM rows found in the provided CSV.")

    # Regret computation at eval granularity: utility = perf - wtp * total_cost
    try:
        if not llms.empty:
            evals = sorted(set(df["eval_name"].unique()))
            import numpy as np

            def regret_at_wtp(w: float):
                c_subset = df[(df["model_name"] == "compitum") & (df.get("willingness_to_pay", w) == w)]
                r_list = []
                wins = 0
                cost_deltas = []
                base_regrets: Dict[str, List[float]] = {}
                for ev in evals:
                    cv = c_subset[c_subset["eval_name"] == ev]
                    if cv.empty:
                        continue
                    c_perf = float(cv["performance"].mean())
                    c_cost = float(cv["total_cost"].mean())
                    c_util = c_perf - w * c_cost
                    lv = llms[llms["eval_name"] == ev].copy()
                    if lv.empty:
                        continue
                    lv["utility"] = lv["performance"] - w * lv["total_cost"]
                    idxmax = lv["utility"].idxmax()
                    best_util = float(lv.loc[idxmax, "utility"])
                    best_cost = float(lv.loc[idxmax, "total_cost"])
                    r_list.append(max(0.0, best_util - c_util))
                    if c_util >= best_util - 1e-12:
                        wins += 1
                        cost_deltas.append(c_cost - best_cost)
                    for name, util in lv.groupby("model_name")["utility"].mean().items():
                        base_regrets.setdefault(str(name), []).append(max(0.0, best_util - float(util)))
                if not r_list:
                    return None
                regrets_by_model = {k: float(pd.Series(v).mean()) for k, v in base_regrets.items()} if base_regrets else {}
                return {
                    "mean_regret": float(np.mean(r_list)),
                    "p95_regret": float(np.percentile(r_list, 95)),
                    "win_rate": float(wins / max(1, len(evals))),
                    "avg_cost_delta_on_wins": float(np.mean(cost_deltas)) if cost_deltas else None,
                    "regrets_by_model": regrets_by_model,
                }

            if wtp_list and len(wtp_list) > 1:
                candidates = [regret_at_wtp(w) for w in wtp_list]
                pairs = [(w, m) for w, m in zip(wtp_list, candidates) if m]
                if pairs:
                    # pick WTP with minimum mean regret
                    w_best, m_best = min(pairs, key=lambda t: t[1]["mean_regret"])
                    metrics.mean_regret = m_best["mean_regret"]
                    metrics.p95_regret = m_best["p95_regret"]
                    metrics.win_rate = m_best["win_rate"]
                    metrics.avg_cost_delta_on_wins = m_best["avg_cost_delta_on_wins"]
                    metrics.notes.append(f"Regret computed at best WTP={w_best}")
                    rbm = dict(m_best.get("regrets_by_model", {}))
                    rbm["compitum"] = float(metrics.mean_regret)
                    metrics.regrets_by_model = rbm
            else:
                m = regret_at_wtp(wtp)
                if m:
                    metrics.mean_regret = m["mean_regret"]
                    metrics.p95_regret = m["p95_regret"]
                    metrics.win_rate = m["win_rate"]
                    metrics.avg_cost_delta_on_wins = m["avg_cost_delta_on_wins"]
                    rbm = dict(m.get("regrets_by_model", {}))
                    rbm["compitum"] = float(metrics.mean_regret)
                    metrics.regrets_by_model = rbm
    except Exception as e:
        metrics.notes.append(f"Regret computation skipped: {e}")

    # (Intentionally keep report simple: do not compute matched-performance savings by default.)
    return metrics


def build_html_report(
    out_path: Path,
    test_summary: Dict[str, str],
    rb_files: List[Path],
    compitum_file: Optional[Path],
    metrics: Optional[MetricsSummary] = None,
    run_meta: Optional[Dict[str, Dict]] = None,
) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    # Simple bar charts for averages if metrics available
    charts = {}
    # Prepare optional summary and glossary sections for non-ML readers
    topline_html = ""
    glossary_html = ""
    fixed_ci_html = ""
    per_task_html = ""
    ablation_html = ""
    if metrics:
        # Performance chart
        labels = ["compitum"] + list(metrics.llm_perf.keys())
        values = [metrics.compitum_perf] + [metrics.llm_perf[k] for k in metrics.llm_perf.keys()]
        # Sort baselines by performance descending for readability (keep compitum first)
        if len(labels) > 1:
            pairs = list(zip(labels[1:], values[1:]))
            pairs.sort(key=lambda t: (t[1] if t[1] == t[1] else -1e9), reverse=True)
            labels = [labels[0]] + [p[0] for p in pairs]
            values = [values[0]] + [p[1] for p in pairs]
        fig, ax = plt.subplots(figsize=(7, 3.2))
        ax.bar(labels, values, color=["#3b82f6"] + ["#9ca3af"] * len(metrics.llm_perf))
        ax.set_title("Average Performance")
        ax.set_ylabel("Accuracy (mean)")
        ax.tick_params(axis='x', rotation=20)
        ax.set_ylim(0, max(1.0, max([v for v in values if v == v] + [1.0])))
        charts["perf"] = _fig_to_data_url(fig)

        # Cost chart
        labels_c = ["compitum"] + list(metrics.llm_cost.keys())
        values_c = [metrics.compitum_cost] + [metrics.llm_cost[k] for k in metrics.llm_cost.keys()]
        if len(labels_c) > 1:
            pairs_c = list(zip(labels_c[1:], values_c[1:]))
            pairs_c.sort(key=lambda t: (t[1] if t[1] == t[1] else 1e9))  # ascending cost
            labels_c = [labels_c[0]] + [p[0] for p in pairs_c]
            values_c = [values_c[0]] + [p[1] for p in pairs_c]
        fig, ax = plt.subplots(figsize=(7, 3.2))
        ax.bar(labels_c, values_c, color=["#10b981"] + ["#9ca3af"] * len(metrics.llm_cost))
        ax.set_title("Average Total Cost (USD)")
        ax.set_ylabel("USD (mean)")
        ax.tick_params(axis='x', rotation=20)
        charts["cost"] = _fig_to_data_url(fig)

        # Regret chart â€“ compare compitum and baselines (baseline best has 0 mean regret by definition)
        if metrics.regrets_by_model:
            items = list(metrics.regrets_by_model.items())
            items = [(k, (v if v == v else 0.0)) for k, v in items]
            comp_item = [(k, v) for k, v in items if k == "compitum"]
            base_items = [(k, v) for k, v in items if k != "compitum"]
            base_items.sort(key=lambda t: t[1])
            labels_r = [comp_item[0][0]] + [k for k, _ in base_items] if comp_item else [k for k, _ in base_items]
            values_r = [comp_item[0][1]] + [v for _, v in base_items] if comp_item else [v for _, v in base_items]
            colors_r = ["#ef4444"] + ["#9ca3af"] * (len(values_r) - 1) if comp_item else ["#9ca3af"] * len(values_r)
            fig, ax = plt.subplots(figsize=(7, 3.2))
            ax.bar(labels_r, values_r, color=colors_r)
            ax.set_title("Mean Regret vs Baseline Best (selected/best WTP)")
            ax.set_ylabel("Utility gap (lower is better)")
            ax.tick_params(axis='x', rotation=20)
            charts["regret"] = _fig_to_data_url(fig)

        # Frontier-like scatter (avg cost vs avg performance)
        try:
            labels = ["compitum"] + list(metrics.llm_perf.keys())
            xs = [metrics.compitum_cost] + [metrics.llm_cost[k] for k in metrics.llm_cost.keys()]
            ys = [metrics.compitum_perf] + [metrics.llm_perf[k] for k in metrics.llm_perf.keys()]
            fig, ax = plt.subplots(figsize=(7, 3.2))
            for i, (x, y) in enumerate(zip(xs, ys)):
                ax.scatter([x], [y], s=40, c=["#22c55e"] if i == 0 else ["#60a5fa"], label=labels[i] if i < 6 else None)
            ax.set_xlabel("Avg Total Cost (USD)")
            ax.set_ylabel("Avg Performance")
            ax.set_title("Average Cost vs Performance (lower cost, higher performance is better)")
            # Avoid duplicate legend entries
            handles, leg_labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(handles[:min(6,len(handles))], leg_labels[:min(6,len(handles))], loc="best")
            charts["frontier"] = _fig_to_data_url(fig)
        except Exception:
            pass
        # Build Topline Takeaways using computed metrics
        best_perf_name = None
        best_perf_val = None
        if metrics.llm_perf:
            best_perf_name, best_perf_val = max(metrics.llm_perf.items(), key=lambda kv: kv[1])
        best_cost_name = None
        best_cost_val = None
        if metrics.llm_cost:
            best_cost_name, best_cost_val = min(metrics.llm_cost.items(), key=lambda kv: kv[1])
        topline_html = f"""
        <div class=\"card\">
          <h2>Topline Takeaways</h2>
          <ul>
            <li><b>Win rate</b> (share of evaluations where Compitumâ€™s utility â‰¥ best baseline): {'' if metrics.win_rate is None else f'{metrics.win_rate*100:.1f}%'}.</li>
            <li><b>Mean regret</b> (utility gap to best baseline at selected/best WTP): {'' if metrics.mean_regret is None else f'{metrics.mean_regret:.6f}'}. Lower is better.</li>
            <li><b>Avg cost delta on wins</b> (Compitum âˆ’ best baseline cost on wins): {'' if metrics.avg_cost_delta_on_wins is None else f'{metrics.avg_cost_delta_on_wins:.6f}'} USD.</li>
            <li><b>Average performance</b>: Compitum {metrics.compitum_perf:.4f}{(' vs ' + best_perf_name + ' ' + f'{best_perf_val:.4f}' if best_perf_name else '')}.</li>
            <li><b>Average cost</b>: Compitum {metrics.compitum_cost:.6f}{(' vs ' + best_cost_name + ' ' + f'{best_cost_val:.6f}' if best_cost_name else '')} USD.</li>
          </ul>
        </div>
        """
    # Fixed-WTP CI table if prior analysis exists
    try:
        root = out_path.resolve().parents[1]
        fixed_json = root / 'reports' / 'fixed_wtp_summary.json'
        if fixed_json.exists():
            data = json.loads(fixed_json.read_text(encoding='utf-8'))
            rows = []
            # keys may be strings; normalize to float for sorting
            parsed = {}
            for k,v in data.items():
                try:
                    parsed[float(k)] = v
                except Exception:
                    pass
            for k in sorted(parsed.keys()):
                ci = parsed[k]
                mr = ci.get('mean_regret', [float('nan')]*3)
                wr = ci.get('win_rate', [float('nan')]*3)
                cd = ci.get('avg_cost_delta_on_wins', [float('nan')]*3)
                rows.append(
                    f"<tr><td>{k:.2f}</td>"
                    f"<td>{mr[0]:.6f} [{mr[1]:.6f}, {mr[2]:.6f}]</td>"
                    f"<td>{wr[0]*100:.1f}% [{wr[1]*100:.1f}%, {wr[2]*100:.1f}%]</td>"
                    f"<td>{cd[0]:.6f} [{cd[1]:.6f}, {cd[2]:.6f}]</td></tr>"
                )
            if rows:
                fixed_ci_html = (
                    "<div class=\"card\"><h2>Fixed WTP Analysis (95% CI)</h2>"
                    "<table><tr><th>WTP</th><th>Mean Regret</th><th>Win Rate</th><th>Avg Cost Î” (wins)</th></tr>"
                    + "".join(rows) + "</table></div>"
                )
    except Exception:
        pass

    # Per-task summary at WTP=1.0
    try:
        if compitum_file:
            import pandas as _pd
            df_task = _pd.read_csv(compitum_file)
            bdf = df_task[df_task["model_name"] != "compitum"].copy()
            cdf = df_task[df_task["model_name"] == "compitum"].copy()
            w = 1.0
            rows = []
            for ev in sorted(set(df_task["eval_name"].astype(str))):
                b = bdf[bdf["eval_name"] == ev]
                c = cdf[cdf["eval_name"] == ev]
                if b.empty or c.empty:
                    continue
                if "willingness_to_pay" in c.columns:
                    c = c[c["willingness_to_pay"] == w] or c
                c_perf = float(c["performance"].mean())
                c_cost = float(c["total_cost"].mean())
                c_util = c_perf - w * c_cost
                b_util = (b["performance"] - w * b["total_cost"]).astype(float)
                idx = int(b_util.idxmax())
                best_util = float(b_util.loc[idx])
                regret = max(0.0, best_util - c_util)
                win = (c_util >= best_util - 1e-12)
                rows.append((ev, regret, 100.0 if win else 0.0))
            if rows:
                top = sorted(rows, key=lambda t: t[1])[:10]
                per_task_html = (
                    "<div class=\"card\"><h2>Perâ€‘Task Summary (WTP = 1.0)</h2>"
                    "<table><tr><th>Eval Name</th><th>Mean Regret</th><th>Win (Utility â‰¥ Best)</th></tr>"
                    + "".join(f"<tr><td>{ev}</td><td>{reg:.6f}</td><td>{wr:.0f}%</td></tr>" for ev, reg, wr in top)
                    + "</table><p class=\"muted\">Top 10 tasks by lowest regret shown.</p></div>"
                )
    except Exception:
        pass

    # Ablation summary table (if available)
    try:
        ab_path = out_path.resolve().parents[1] / 'reports' / 'ablation_summary.json'
        if ab_path.exists():
            data = json.loads(ab_path.read_text(encoding='utf-8'))
            rows = []
            def _wtp_sort_key(k: str) -> float:
                try:
                    return float(str(k).split('=')[-1])
                except Exception:
                    return 0.0
            for wtp in sorted(data.keys(), key=_wtp_sort_key):
                entries = data[wtp]
                for model in ("compitum", "knn", "mlp", "cascading"):
                    stats = entries.get(model)
                    if not stats:
                        continue
                    mr = stats.get('mean_regret', float('nan'))
                    rows.append(f"<tr><td>{wtp}</td><td>{model}</td><td>{mr:.6f}</td></tr>")
            if rows:
                ablation_html = (
                    "<div class=\"card\"><h2>Ablation Summary (Fixed WTP)</h2>"
                    "<table><tr><th>WTP</th><th>Model</th><th>Mean Regret</th></tr>"
                    + "".join(rows) + "</table></div>"
                )
    except Exception:
        pass

    # Glossary card
    glossary_html = """
    <div class=\"card\">
      <h2>Glossary</h2>
      <ul>
        <li><b>Performance</b>: Average task accuracy across the evaluation set (higher is better).</li>
        <li><b>Total cost</b>: Average token and compute cost in USD (lower is better).</li>
        <li><b>Willingness to Pay (WTP)</b>: How much performance is worth relative to cost; utility = performance âˆ’ WTP Ã— cost.</li>
        <li><b>Utility</b>: Single-number trade-off of performance and cost at a chosen WTP.</li>
        <li><b>Mean regret</b>: Average utility gap to the best baseline at the chosen WTP (lower is better).</li>
        <li><b>Win rate</b>: Share of evaluations where Compitumâ€™s utility â‰¥ best baseline utility.</li>
      </ul>
    </div>
    """

    style = """
    body{font-family:system-ui,-apple-system,Segoe UI,Roboto,Ubuntu,Arial,sans-serif;max-width:1000px;margin:32px auto;padding:0 16px;color:#111827}
    h1{font-size:22px;margin:0 0 12px}
    h2{font-size:18px;margin:24px 0 8px}
    .card{border:1px solid #e5e7eb;border-radius:8px;padding:16px;margin:16px 0}
    table{border-collapse:collapse;width:100%}
    th,td{border:1px solid #e5e7eb;padding:8px;text-align:left}
    .muted{color:#6b7280}
    .grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}
    img{max-width:100%}
    code{background:#f3f4f6;padding:2px 4px;border-radius:4px}
    """

    rb_list = "".join(f"<li><code>{p}</code></li>" for p in rb_files)
    compitum_link = f"<li><code>{compitum_file}</code></li>" if compitum_file else "<li class='muted'>none</li>"

    # Step status card
    step_html = ""
    if run_meta:
        def _fmt(step):
            if not step:
                return ("N/A", "", "", "")
            return (
                step.get("returncode"),
                "YES" if step.get("timed_out") else "NO",
                step.get("duration_sec"),
                step.get("timeout_sec"),
            )
        tests_rc, tests_to, tests_dt, tests_cap = _fmt(run_meta.get("tests"))
        rb_rc, rb_to, rb_dt, rb_cap = _fmt(run_meta.get("routerbench"))
        comp_rc, comp_to, comp_dt, comp_cap = _fmt(run_meta.get("compitum"))
        step_html = f"""
        <div class="card">
          <h2>Step Status</h2>
          <table>
            <tr><th>Step</th><th>Return Code</th><th>Timed Out</th><th>Duration (s)</th><th>Timeout Cap (s)</th></tr>
            <tr><td>Unit Tests</td><td>{tests_rc}</td><td>{tests_to}</td><td>{tests_dt}</td><td>{tests_cap}</td></tr>
            <tr><td>RouterBench</td><td>{rb_rc}</td><td>{rb_to}</td><td>{rb_dt}</td><td>{rb_cap}</td></tr>
            <tr><td>Compitum</td><td>{comp_rc}</td><td>{comp_to}</td><td>{comp_dt}</td><td>{comp_cap}</td></tr>
          </table>
          <p class="muted">Full machine-readable log stored alongside this report as JSON.</p>
        </div>
        """

    metrics_html = ""
    if metrics:
        metrics_html = f"""
        <div class="grid">
          <div class="card"><img src="{charts.get('perf','')}" alt="Average Performance"></div>
          <div class="card"><img src="{charts.get('cost','')}" alt="Average Cost"></div>
        </div>
        {('<div class="card"><img src="' + charts.get('regret','') + '" alt="Mean Regret"></div>') if charts.get('regret') else ''}
        {('<div class="card"><img src="' + charts.get('frontier','') + '" alt="Avg Cost vs Performance"></div>') if charts.get('frontier') else ''}
        <div class="card">
          <h2>Numerical Summary</h2>
          <table>
            <tr><th>Model</th><th>Avg Performance</th><th>Avg Total Cost</th></tr>
            <tr><td>compitum</td><td>{metrics.compitum_perf:.4f}</td><td>{metrics.compitum_cost:.6f}</td></tr>
            {''.join(f"<tr><td>{name}</td><td>{metrics.llm_perf.get(name,float('nan')):.4f}</td><td>{metrics.llm_cost.get(name,float('nan')):.6f}</td></tr>" for name in metrics.llm_perf.keys())}
          </table>
          <h2>Regret & Wins (WTP-selected)</h2>
          <table>
            <tr><th>Mean Regret</th><th>P95 Regret</th><th>Win Rate</th><th>Avg Cost Delta on Wins</th></tr>
            <tr>
              <td>{'' if metrics.mean_regret is None else f'{metrics.mean_regret:.6f}'}</td>
              <td>{'' if metrics.p95_regret is None else f'{metrics.p95_regret:.6f}'}</td>
              <td>{'' if metrics.win_rate is None else f'{metrics.win_rate*100:.1f}%'} </td>
              <td>{'' if metrics.avg_cost_delta_on_wins is None else f'{metrics.avg_cost_delta_on_wins:.6f}'}</td>
            </tr>
          </table>
          {('<p class="muted">' + ' '.join(metrics.notes) + '</p>') if metrics.notes else ''}
        </div>
        """

    html = f"""
    <html><head><meta charset="utf-8"><title>Compitum Report</title>
    <style>{style}</style></head>
    <body>
      <h1>Compitum Test & Benchmark Report</h1>
      <p class="muted">Generated {ts}</p>

      <div class="card">
        <h2>Overview</h2>
        <p>This report summarizes Compitumâ€™s test and benchmark results alongside common baselines.</p>
        <ul>
          <li><b>Performance</b>: average task accuracy (higher is better).</li>
          <li><b>Total cost</b>: average token-compute cost in USD (lower is better).</li>
          <li><b>Utility</b>: performance âˆ’ WTP Ã— cost, where <b>WTP</b> (willingness to pay) scales the cost penalty.</li>
          <li><b>Mean regret</b>: average gap to the best baseline utility at a selected WTP (lower is better).</li>
          <li><b>Win rate</b>: fraction of evaluations where Compitumâ€™s utility â‰¥ best baseline utility at the selected WTP.</li>
        </ul>
        <p class="muted">Charts: bars include Compitum (blue/green) and baselines (gray). The scatter shows average cost vs performance for all models.</p>
      </div>

      {topline_html}

      <div class="card">
        <h2>Unit Tests</h2>
        <pre>{test_summary.get('stdout','').strip()}</pre>
        <p class="muted">Exit code: {test_summary.get('returncode','')}</p>
      </div>

      {step_html}

      <div class="card">
        <h2>RouterBench Artifacts</h2>
        <ul>
          {rb_list}
        </ul>
        <h3>Compitum Artifacts</h3>
        <ul>
          {compitum_link}
        </ul>
      </div>

      {metrics_html}

      {ablation_html}

      {glossary_html}
    </body></html>
    """
    out_path.write_text(html, encoding="utf-8")
    return out_path
