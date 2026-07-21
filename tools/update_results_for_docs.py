from pathlib import Path
import json


def format_ci_mid_low_high(triple):
    # Data stored as [mid, low, high]
    mid, low, high = triple
    return f"{mid:.6f} [{low:.6f}, {high:.6f}]"


def main() -> int:
    json_src = Path("reports") / "fixed_wtp_summary.json"
    md_src = Path("reports") / "fixed_wtp_summary.md"
    dst = Path("docs") / "Results-Fixed-WTP.md"

    header = "---\ntitle: Results - Fixed WTP (Auto)\n---\n\n# Fixed-WTP Analysis (95% CI)\n\n"

    if json_src.exists():
        data = json.loads(json_src.read_text(encoding="utf-8", errors="ignore"))
        rows = []
        for wtp_key in sorted(data.keys(), key=lambda x: float(x)):
            entry = data[wtp_key]
            mean_regret = format_ci_mid_low_high(entry.get("mean_regret", [0, 0, 0]))
            # fixed_wtp_ci.py's _ci() returns (mid, low, high) -- indices 0/1/2,
            # same order mean_regret uses above. This previously read index 1
            # (low) as the displayed value and index 0 (the real mid) as the
            # CI's lower bound, invisible only because every win_rate here
            # happens to be 0.0 (low == mid == high == 0).
            win_rate_ci = entry.get("win_rate", [0, 0, 0])
            win_rate = f"{win_rate_ci[0] * 100:.1f}% [{win_rate_ci[1] * 100:.1f}%, {win_rate_ci[2] * 100:.1f}%]"
            acd = entry.get("avg_cost_delta_on_wins")
            if acd is None or any(v is None for v in acd):
                avg_cost_delta = "N/A"
            else:
                avg_cost_delta = format_ci_mid_low_high(acd)
            rows.append((float(wtp_key), mean_regret, win_rate, avg_cost_delta))

        # Build a clean Markdown table
        lines = [
            "| WTP | Mean Regret | Win Rate | Avg Cost Delta (wins) |",
            "|---:|---:|---:|---:|",
        ]
        for wtp, mr, wr, acd in rows:
            lines.append(f"| {wtp:0.2f} | {mr} | {wr} | {acd} |")
        body = "\n".join(lines) + "\n"
        dst.write_text(header + body, encoding="utf-8")
        return 0

    if md_src.exists():
        # Fallback: wrap the upstream MD (may contain odd glyphs)
        content = md_src.read_text(encoding="utf-8", errors="ignore")
        # Strip leading heading from upstream if present
        if content.lstrip().startswith("#"):
            content = "\n".join(
                line for line in content.splitlines() if not line.strip().startswith("#")
            ).lstrip()
        dst.write_text(header + content + "\n", encoding="utf-8")
        return 0

    # Placeholder if neither source exists
    dst.write_text(
        header + "No results found. Run `make peer-review` to generate results.\n",
        encoding="utf-8",
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
