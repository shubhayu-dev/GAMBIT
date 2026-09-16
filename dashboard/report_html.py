"""
Dashboard (docs/10 Week 8: "dashboard + visualizations + experiment report").

Generates a static, self-contained HTML report integrating statistical metrics 
and the LLM's move-by-move telemetry reasoning.
"""

from typing import Dict, List


def _bar(label: str, value: float, max_value: float = 100.0, suffix: str = "%") -> str:
    pct = max(0.0, min(100.0, (value / max_value) * 100)) if value is not None else 0
    display = f"{value:.1f}{suffix}" if value is not None else "n/a"
    return f"""
    <div class="bar-row">
      <span class="bar-label">{label}</span>
      <div class="bar-track"><div class="bar-fill" style="width:{pct:.1f}%"></div></div>
      <span class="bar-value">{display}</span>
    </div>"""


def build_report_html(agent_name: str, profile: Dict, weakness_dim: str, weakness_desc: str,
                       ranked_hypotheses: List[Dict], intervention_result: Dict,
                       deviations: List[str], llm_diagnosis: Dict = None) -> str:
    dim_rows = "".join(
        _bar(name.replace("_proxy", "*").replace("_", " ").title(), profile[key]["decision_accuracy_pct"])
        for name, key in [
            ("Opening", "opening"), ("Middlegame", "middlegame"), ("Endgame", "endgame"),
            ("Tactical*", "tactical_proxy"), ("Positional*", "positional_proxy"), ("Defensive*", "defensive_proxy"),
        ]
    )

    hyp_rows = "".join(f"""
      <tr class="{'winner' if i == 0 else ''}">
        <td>{h['hypothesis']}</td>
        <td>{h['label_a']} &rarr; {h['label_b']}</td>
        <td>{h['mean_abs_regret_a']} &rarr; {h['mean_abs_regret_b']}</td>
        <td>{h['improvement_pct']}%</td>
        <td>{h['p_value'] if h['p_value'] is not None else '&mdash;'}</td>
      </tr>""" for i, h in enumerate(ranked_hypotheses))

    before = intervention_result["before"]["mean_abs_regret"]
    after = intervention_result["after"]["mean_abs_regret"]
    improvement = intervention_result["improvement_pct"]

    deviation_items = "".join(f"<li>{d}</li>" for d in deviations)
    overall_acc = profile["overall"]["decision_accuracy_pct"]

    # --- Build the LLM Diagnosis Section ---
    llm_section = ""
    if llm_diagnosis:
        interp = llm_diagnosis.get("interpretation", "No interpretation available.")
        best_hyp = llm_diagnosis.get("best_supported_hypothesis", "None")
        
        cases_html = ""
        for cs in llm_diagnosis.get("case_study_analysis", []):
            fen = cs.get("position", "Unknown")
            div = cs.get("divergence_analysis", "")
            diag = cs.get("diagnosis", "")
            cases_html += f"""
            <div class="case-study">
              <div class="fen">FEN: {fen}</div>
              <div class="analysis"><strong>Divergence:</strong> {div}</div>
              <div class="analysis"><strong>Diagnosis:</strong> {diag}</div>
            </div>
            """
            
        llm_section = f"""
        <h2>03 — Gemini Glass-Box Diagnosis</h2>
        <div class="llm-box">
          <p class="interp"><strong>Interpretation:</strong> {interp}</p>
          <p class="interp"><strong>Top Supported Hypothesis:</strong> {best_hyp}</p>
          {cases_html}
        </div>
        """

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>GAMBIT Report — {agent_name}</title>
<style>
  :root {{
    --bg: #17140f;
    --panel: #1f1b14;
    --line: #3a3327;
    --ink: #ece4d3;
    --ink-dim: #a89b83;
    --gold: #c9a227;
    --good: #7a9b6e;
    --bad: #b5674f;
    --ai: #8e7cc3;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    background: var(--bg);
    color: var(--ink);
    font-family: Georgia, 'Iowan Old Style', serif;
    max-width: 780px;
    margin: 0 auto;
    padding: 48px 24px 80px;
    line-height: 1.5;
  }}
  .mono {{ font-family: 'SF Mono', 'JetBrains Mono', Menlo, monospace; }}
  h1 {{
    font-size: 28px;
    margin: 0 0 4px;
    letter-spacing: 0.2px;
  }}
  .subtitle {{ color: var(--ink-dim); font-size: 14px; margin-bottom: 40px; }}
  .subtitle span.mono {{ color: var(--gold); }}
  h2 {{
    font-size: 15px;
    text-transform: none;
    color: var(--gold);
    border-bottom: 1px solid var(--line);
    padding-bottom: 8px;
    margin: 40px 0 20px;
    font-family: 'SF Mono', 'JetBrains Mono', Menlo, monospace;
    letter-spacing: 0.3px;
  }}
  .headline-score {{
    display: flex;
    align-items: baseline;
    gap: 16px;
    background: var(--panel);
    border: 1px solid var(--line);
    padding: 24px 28px;
    border-radius: 2px;
  }}
  .headline-score .num {{ font-size: 52px; color: var(--gold); font-family: 'SF Mono', monospace; }}
  .headline-score .label {{ color: var(--ink-dim); font-size: 14px; }}
  .bar-row {{ display: flex; align-items: center; gap: 12px; margin-bottom: 12px; }}
  .bar-label {{ width: 120px; font-size: 13px; color: var(--ink-dim); flex-shrink: 0; }}
  .bar-track {{ flex: 1; height: 10px; background: var(--panel); border: 1px solid var(--line); }}
  .bar-fill {{ height: 100%; background: var(--gold); }}
  .bar-value {{ width: 56px; text-align: right; font-family: monospace; font-size: 13px; color: var(--ink); }}
  .weakness-box {{
    background: var(--panel);
    border-left: 3px solid var(--bad);
    padding: 16px 20px;
    font-size: 14px;
  }}
  .llm-box {{
    background: var(--panel);
    border-left: 3px solid var(--ai);
    padding: 20px;
    font-size: 14px;
  }}
  .llm-box .interp {{ margin-top: 0; margin-bottom: 8px; }}
  .case-study {{ margin-top: 20px; padding-top: 16px; border-top: 1px dashed var(--line); }}
  .case-study .fen {{ font-family: monospace; color: var(--gold); margin-bottom: 8px; font-size: 12px; }}
  .case-study .analysis {{ margin-bottom: 6px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  th, td {{ text-align: left; padding: 8px 10px; border-bottom: 1px solid var(--line); }}
  th {{ color: var(--ink-dim); font-weight: normal; font-family: monospace; font-size: 12px; }}
  tr.winner td:first-child {{ color: var(--gold); }}
  .validation {{ display: flex; gap: 24px; align-items: center; }}
  .validation .col {{ flex: 1; background: var(--panel); border: 1px solid var(--line); padding: 18px; text-align: center; }}
  .validation .col .n {{ font-size: 30px; font-family: monospace; }}
  .validation .col.before .n {{ color: var(--bad); }}
  .validation .col.after .n {{ color: var(--good); }}
  .validation .arrow {{ font-size: 24px; color: var(--ink-dim); }}
  .improve {{ text-align: center; margin-top: 12px; font-size: 14px; color: var(--good); }}
  footer {{ margin-top: 56px; font-size: 12px; color: var(--ink-dim); border-top: 1px solid var(--line); padding-top: 16px; }}
  footer ul {{ padding-left: 18px; }}
</style>
</head>
<body>

  <h1>GAMBIT</h1>
  <div class="subtitle">Agent Evaluation & Diagnostic Report &mdash; <span class="mono">{agent_name}</span></div>

  <div class="headline-score">
    <div class="num">{overall_acc:.0f}%</div>
    <div class="label">overall decision accuracy<br>({profile['overall']['n']} positions, |regret| &le; 0.5 pawns)</div>
  </div>

  <h2>01 — Capability Profile</h2>
  {dim_rows}

  <h2>02 — Primary Weakness</h2>
  <div class="weakness-box">{weakness_desc}</div>

  {llm_section}

  <h2>04 — Diagnostic Experiments</h2>
  <table>
    <tr><th>Hypothesis</th><th>Condition A &rarr; B</th><th>Mean |regret| A &rarr; B</th><th>Improvement</th><th>p-value</th></tr>
    {hyp_rows}
  </table>
  <p style="font-size:12px;color:var(--ink-dim)">Ranked by improvement magnitude; the top row is the best-supported cause. Small sample sizes here (n shown in each hypothesis's underlying run) mean p-values should be read as directional, not conclusive.</p>

  <h2>05 — Intervention: Adaptive Search Depth</h2>
  <div class="validation">
    <div class="col before"><div class="n">{before}</div><div>before (fixed depth)</div></div>
    <div class="arrow">&rarr;</div>
    <div class="col after"><div class="n">{after}</div><div>after (adaptive depth)</div></div>
  </div>
  <div class="improve">{improvement}% reduction in mean |regret| on {intervention_result['n_positions']} held-out positions (never used during diagnosis)</div>

  <footer>
    <strong>Noted deviations from the original spec (no network access in this build environment):</strong>
    <ul>{deviation_items}</ul>
  </footer>

</body>
</html>"""