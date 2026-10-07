"""
Dashboard (docs/10 Week 8: "dashboard + visualizations + experiment report").

Generates a static, self-contained HTML report. Two audiences at once:
  - a normal person: plain-English verdicts, explanations, and chess board
    pictures of the actual blunders found
  - an analyst: every underlying number, the diagnostic experiment table,
    the held-out validation result, and a glossary for anything jargon-y

No JS, no external assets (fonts/CDNs) -- everything, including the chess
board diagrams, is inline SVG generated server-side by chess_board_svg.py,
so the single HTML file works offline and never breaks if a CDN goes down.
"""

from typing import Dict, List, Optional

from chess_board_svg import render_board_svg

# ---------------------------------------------------------------------------
# Plain-English translation helpers
# ---------------------------------------------------------------------------

def _verdict(overall_acc: Optional[float]) -> str:
    if overall_acc is None:
        return "Not enough data yet"
    if overall_acc >= 90:
        return "Strong — rarely picks a worse move than the reference"
    if overall_acc >= 75:
        return "Solid, but inconsistent"
    if overall_acc >= 50:
        return "Shaky — gets it right about half the time"
    return "Struggling — wrong more often than right"


def _regret_in_words(mean_abs_regret: Optional[float]) -> str:
    if mean_abs_regret is None:
        return "no data"
    if mean_abs_regret < 0.3:
        return "a rounding error — basically no cost"
    if mean_abs_regret < 1.0:
        return "roughly a pawn's worth of value, on average, every time it picks the wrong move"
    if mean_abs_regret < 3.0:
        return "more than a pawn, sometimes close to giving away a minor piece (knight/bishop) for nothing"
    return "a serious amount — on an average mistake it's giving away at least a piece's worth of value, sometimes a lot more"


def _dim_blurb(name: str, stats: Dict, overall_acc: Optional[float], is_weakest: bool = False) -> str:
    n, acc = stats.get("n", 0), stats.get("decision_accuracy_pct")
    if not n:
        return "No test positions landed in this bucket."
    if n < 3:
        return f"Only {n} position{'s' if n != 1 else ''} tested here — too few to draw a real conclusion."
    if acc is None:
        return "No data."
    delta = None if overall_acc is None else round(acc - overall_acc, 1)
    if delta is None:
        trend = ""
    elif delta >= 5:
        trend = f" — better than its {overall_acc:.0f}% overall average"
    elif delta <= -5:
        trend = f" — worse than its {overall_acc:.0f}% overall average"
    else:
        trend = " — about in line with its overall average"
    flag = " <strong class='weak-flag'>&larr; weakest measured dimension</strong>" if is_weakest else ""
    return f"{acc:.0f}% correct across {n} positions{trend}.{flag}"


_DIM_LABELS = [
    ("Opening", "opening", "Early-game positions, still mostly developed."),
    ("Middlegame", "middlegame", "The heavy-piece, tactically dense middle of the game."),
    ("Endgame", "endgame", "Simplified positions with few pieces left."),
    ("Tactical*", "tactical_proxy", "Positions with lots of legal replies (sharp, many candidate moves) — a stand-in for "
                                      "'tactical' since we don't have hand-labeled tags yet."),
    ("Positional*", "positional_proxy", "Quieter positions with fewer candidate moves — a stand-in for 'positional'."),
    ("Defensive*", "defensive_proxy", "Positions where the side to move is already worse off and has to defend."),
]


def _bar(label: str, value: Optional[float], max_value: float = 100.0, suffix: str = "%") -> str:
    pct = max(0.0, min(100.0, (value / max_value) * 100)) if value is not None else 0
    display = f"{value:.1f}{suffix}" if value is not None else "n/a"
    return f"""
    <div class="bar-row">
      <span class="bar-label">{label}</span>
      <div class="bar-track"><div class="bar-fill" style="width:{pct:.1f}%"></div></div>
      <span class="bar-value">{display}</span>
    </div>"""


def _fmt(v, suffix="", digits=2, dash="&mdash;"):
    if v is None:
        return dash
    try:
        return f"{v:.{digits}f}{suffix}"
    except (TypeError, ValueError):
        return str(v)


# ---------------------------------------------------------------------------
# Blunder case study rendering
# ---------------------------------------------------------------------------

def _regret_magnitude_phrase(regret: float) -> str:
    """Translates a pawn-unit regret number into a rough material comparison
    a non-chess-player can picture. Clipped regret can run up to 40 (see
    CLIP_PAWNS in experiments/runner.py), which almost always means a missed
    or allowed forced mate rather than literally losing that much material --
    said explicitly rather than implying an impossible material loss."""
    if regret >= 9.0:
        return "an enormous swing — this size of gap usually means a missed (or allowed) forced checkmate, not just lost material"
    if regret >= 6.0:
        return "roughly a rook's worth"
    if regret >= 2.5:
        return "roughly a minor piece (knight/bishop)"
    if regret >= 1.0:
        return "roughly a pawn's worth"
    return "a small but real edge"


def _blunder_card(idx: int, rec: Dict) -> str:
    fen = rec.get("position", "")
    chosen = rec.get("chosen_move")
    reference = rec.get("reference_move")
    regret = rec.get("regret", 0.0)
    phase = rec.get("game_phase", "unknown")
    rating = rec.get("rating")
    match = rec.get("decision_match")
    pv = rec.get("agent_pv_line", "")
    nodes = rec.get("agent_nodes_searched")
    depth = rec.get("agent_depth_reached")

    svg = render_board_svg(fen, chosen_uci=chosen, reference_uci=reference, size=300)

    side_to_move = "White" if fen.split()[1] == "w" else "Black"
    if match:
        plain = (f"The engine played <strong>{chosen}</strong>, which matches the reference move — "
                 "no mistake here, shown as an example of a position it handled correctly.")
    else:
        plain = (f"{side_to_move} to move. The engine played <strong class='bad-move'>{chosen}</strong> "
                 f"(red arrow) instead of the reference move <strong class='good-move'>{reference}</strong> "
                 f"(green arrow). That choice cost about <strong>{regret:.1f} pawns</strong> of value — "
                 f"{_regret_magnitude_phrase(regret)}.")

    rating_str = f", puzzle rated {rating}" if rating else ""
    tag = "CORRECT MOVE EXAMPLE" if match else f"#{idx} — regret {regret:.1f} pawns"
    tag_class = "blunder-tag good" if match else "blunder-tag"
    return f"""
    <div class="blunder-card">
      <div class="board-wrap">{svg}</div>
      <div class="blunder-info">
        <div class="{tag_class}">{tag}</div>
        <p class="plain">{plain}</p>
        <div class="blunder-meta mono">
          phase: {phase}{rating_str}<br>
          engine's planned line: {pv or '&mdash;'}<br>
          searched {nodes if nodes is not None else '?'} nodes to depth {depth if depth is not None else '?'}
        </div>
      </div>
    </div>"""


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build_report_html(agent_name: str, profile: Dict, weakness_dim: str, weakness_desc: str,
                       ranked_hypotheses: List[Dict], intervention_result: Dict,
                       deviations: List[str], llm_diagnosis: Dict = None,
                       blunder_examples: Optional[List[Dict]] = None) -> str:

    overall = profile.get("overall", {})
    overall_acc = overall.get("decision_accuracy_pct")
    overall_n = overall.get("n", 0)
    mean_abs_regret = overall.get("mean_abs_regret")
    regret_stdev = profile.get("consistency", {}).get("regret_stdev")
    mean_time = profile.get("efficiency", {}).get("mean_time_used_s")

    dim_rows = "".join(
        f"""<div class="dim-block{' dim-weakest' if key == weakness_dim else ''}">
              {_bar(label, profile.get(key, {}).get("decision_accuracy_pct"))}
              <p class="dim-blurb">{_dim_blurb(label, profile.get(key, {}), overall_acc, is_weakest=(key == weakness_dim))} <span class="dim-note">{note}</span></p>
            </div>"""
        for label, key, note in _DIM_LABELS
    )

    hyp_rows = "".join(f"""
      <tr class="{'winner' if i == 0 else ''}">
        <td>{h['hypothesis']}</td>
        <td>{h['label_a']} &rarr; {h['label_b']}</td>
        <td>{_fmt(h['mean_abs_regret_a'])} &rarr; {_fmt(h['mean_abs_regret_b'])}</td>
        <td>{_fmt(h['improvement_pct'], '%', 1)}</td>
        <td>{h['p_value'] if h.get('p_value') is not None else '&mdash;'}</td>
      </tr>""" for i, h in enumerate(ranked_hypotheses)) or \
      '<tr><td colspan="5" style="color:var(--ink-dim)">No diagnostic experiment produced a candidate fix.</td></tr>'

    before = intervention_result.get("before", {}).get("mean_abs_regret")
    after = intervention_result.get("after", {}).get("mean_abs_regret")
    improvement = intervention_result.get("improvement_pct")
    status = intervention_result.get("status", "rejected")
    n_held = intervention_result.get("n_positions")

    if status == "applied":
        intervention_plain = (
            f"The fix (<strong>{intervention_result.get('type', 'candidate change')}</strong>) was tested on "
            f"{n_held} positions it had never seen before, and it actually held up there too — mistakes got "
            f"{abs(improvement):.0f}% smaller on average. This one is worth keeping."
        )
    else:
        direction = "didn't hold up" if (improvement is not None and improvement <= 15) else "wasn't tested"
        worse_note = " — it actually made things <em>worse</em> on new positions" if (improvement is not None and improvement < 0) else ""
        intervention_plain = (
            f"A candidate fix (<strong>{intervention_result.get('type', 'none found')}</strong>) looked promising on "
            f"the positions used to find it, but when checked against {n_held or 0} <em>held-out</em> positions it had "
            f"never seen, it {direction}{worse_note}. So it was rejected rather than rolled out — exactly the "
            "point of testing on unseen data before trusting a fix."
        )

    deviation_items = "".join(f"<li>{d}</li>" for d in deviations)

    # --- Blunder case studies (the chess board visuals) ---
    blunder_section = ""
    if blunder_examples:
        cards = "".join(_blunder_card(i + 1, rec) for i, rec in enumerate(blunder_examples))
        blunder_section = f"""
        <h2>03 — What It Actually Got Wrong (With the Board)</h2>
        <p class="section-intro">The positions below are the actual worst mistakes found in testing, ranked by how
        costly the wrong move was. Red arrow = what the engine played. Green arrow = what it should have played.</p>
        <div class="blunder-grid">{cards}</div>
        """

    # --- Optional LLM diagnosis section ---
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
            </div>"""
        llm_section = f"""
        <h2>04 — AI-Assisted Diagnosis</h2>
        <div class="llm-box">
          <p class="interp"><strong>Interpretation:</strong> {interp}</p>
          <p class="interp"><strong>Top Supported Hypothesis:</strong> {best_hyp}</p>
          {cases_html}
        </div>"""

    plain_summary = (
        (f"We tested this engine on {overall_n} chess positions where we know the objectively correct (or close to "
         f"correct) move. It picked the right move <strong>{overall_acc:.0f}%</strong> of the time. "
         if overall_acc is not None else "We tested this engine, but there isn't enough data yet for a headline number. ")
        + (f"When it does get it wrong, the mistake costs {_regret_in_words(mean_abs_regret)}. " if mean_abs_regret is not None else "")
        + weakness_desc
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>GAMBIT Report — {agent_name}</title>
<style>
  :root {{
    --bg: #17140f;
    --panel: #1f1b14;
    --panel2: #241f16;
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
    max-width: 920px;
    margin: 0 auto;
    padding: 48px 24px 80px;
    line-height: 1.55;
  }}
  .mono {{ font-family: 'SF Mono', 'JetBrains Mono', Menlo, monospace; }}
  h1 {{ font-size: 28px; margin: 0 0 4px; letter-spacing: 0.2px; }}
  .subtitle {{ color: var(--ink-dim); font-size: 14px; margin-bottom: 32px; }}
  .subtitle span.mono {{ color: var(--gold); }}
  h2 {{
    font-size: 15px;
    color: var(--gold);
    border-bottom: 1px solid var(--line);
    padding-bottom: 8px;
    margin: 44px 0 16px;
    font-family: 'SF Mono', 'JetBrains Mono', Menlo, monospace;
    letter-spacing: 0.3px;
  }}
  .section-intro {{ color: var(--ink-dim); font-size: 14px; margin: -6px 0 20px; max-width: 70ch; }}

  .plain-summary {{
    background: var(--panel);
    border: 1px solid var(--line);
    border-left: 3px solid var(--gold);
    padding: 20px 24px;
    font-size: 16px;
    border-radius: 2px;
  }}
  .plain-summary strong {{ color: var(--gold); }}

  .card-row {{ display: flex; gap: 14px; margin-top: 20px; flex-wrap: wrap; }}
  .stat-card {{
    flex: 1 1 150px;
    background: var(--panel);
    border: 1px solid var(--line);
    padding: 16px 18px;
    border-radius: 2px;
  }}
  .stat-card .n {{ font-size: 26px; font-family: monospace; color: var(--gold); }}
  .stat-card .label {{ color: var(--ink-dim); font-size: 12px; margin-top: 4px; }}

  .dim-block {{ margin-bottom: 18px; }}
  .bar-row {{ display: flex; align-items: center; gap: 12px; margin-bottom: 4px; }}
  .bar-label {{ width: 110px; font-size: 13px; color: var(--ink-dim); flex-shrink: 0; }}
  .bar-track {{ flex: 1; height: 10px; background: var(--panel); border: 1px solid var(--line); }}
  .bar-fill {{ height: 100%; background: var(--gold); }}
  .bar-value {{ width: 56px; text-align: right; font-family: monospace; font-size: 13px; color: var(--ink); }}
  .dim-blurb {{ margin: 2px 0 0 122px; font-size: 13px; color: var(--ink-dim); max-width: 64ch; }}
  .dim-note {{ display: block; font-size: 11px; opacity: 0.75; margin-top: 2px; }}
  .dim-weakest .bar-label {{ color: var(--bad); }}
  .dim-weakest .bar-fill {{ background: var(--bad); }}
  .weak-flag {{ color: var(--bad); font-weight: normal; }}

  .weakness-box {{ background: var(--panel); border-left: 3px solid var(--bad); padding: 16px 20px; font-size: 14px; }}
  .llm-box {{ background: var(--panel); border-left: 3px solid var(--ai); padding: 20px; font-size: 14px; }}
  .llm-box .interp {{ margin-top: 0; margin-bottom: 8px; }}
  .case-study {{ margin-top: 20px; padding-top: 16px; border-top: 1px dashed var(--line); }}
  .case-study .fen {{ font-family: monospace; color: var(--gold); margin-bottom: 8px; font-size: 12px; }}
  .case-study .analysis {{ margin-bottom: 6px; }}

  table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  th, td {{ text-align: left; padding: 8px 10px; border-bottom: 1px solid var(--line); }}
  th {{ color: var(--ink-dim); font-weight: normal; font-family: monospace; font-size: 12px; }}
  tr.winner td:first-child {{ color: var(--gold); }}

  .validation {{ display: flex; gap: 24px; align-items: center; flex-wrap: wrap; }}
  .validation .col {{ flex: 1; min-width: 140px; background: var(--panel); border: 1px solid var(--line); padding: 18px; text-align: center; }}
  .validation .col .n {{ font-size: 30px; font-family: monospace; }}
  .validation .col.before .n {{ color: var(--bad); }}
  .validation .col.after .n {{ color: var(--good); }}
  .validation .arrow {{ font-size: 24px; color: var(--ink-dim); }}
  .plain-callout {{ margin-top: 14px; font-size: 14px; background: var(--panel2); border: 1px solid var(--line); padding: 14px 18px; border-radius: 2px; }}

  .blunder-grid {{ display: flex; flex-direction: column; gap: 20px; }}
  .blunder-card {{
    display: flex; gap: 20px; background: var(--panel); border: 1px solid var(--line);
    padding: 18px; border-radius: 2px; flex-wrap: wrap;
  }}
  .board-wrap {{ flex: 0 0 auto; }}
  .board-wrap svg {{ display: block; width: 260px; height: 260px; }}
  .blunder-info {{ flex: 1 1 260px; min-width: 220px; }}
  .blunder-tag {{
    font-family: monospace; font-size: 12px; color: var(--bad); text-transform: uppercase;
    letter-spacing: 0.4px; margin-bottom: 8px;
  }}
  .blunder-tag.good {{ color: var(--good); }}
  .plain {{ font-size: 14px; margin: 0 0 12px; }}
  .plain .bad-move {{ color: var(--bad); }}
  .plain .good-move {{ color: var(--good); }}
  .blunder-meta {{ font-size: 11px; color: var(--ink-dim); line-height: 1.6; }}

  .glossary dt {{ color: var(--gold); font-family: monospace; font-size: 13px; margin-top: 12px; }}
  .glossary dd {{ margin: 2px 0 0; font-size: 13px; color: var(--ink-dim); }}

  footer {{ margin-top: 56px; font-size: 12px; color: var(--ink-dim); border-top: 1px solid var(--line); padding-top: 16px; }}
  footer ul {{ padding-left: 18px; }}
</style>
</head>
<body>

  <h1>GAMBIT</h1>
  <div class="subtitle">Agent Evaluation & Diagnostic Report &mdash; <span class="mono">{agent_name}</span></div>

  <h2>00 — In Plain English</h2>
  <div class="plain-summary">{plain_summary}</div>

  <div class="card-row">
    <div class="stat-card"><div class="n">{_fmt(overall_acc, '%', 0)}</div><div class="label">decision accuracy ({overall_n} positions)</div></div>
    <div class="stat-card"><div class="n">{_fmt(mean_abs_regret)}</div><div class="label">avg. mistake size (pawns)</div></div>
    <div class="stat-card"><div class="n">{_fmt(regret_stdev)}</div><div class="label">consistency (lower = steadier)</div></div>
    <div class="stat-card"><div class="n">{_fmt(mean_time, 's', 2)}</div><div class="label">avg. thinking time / move</div></div>
  </div>

  <h2>01 — Capability Profile</h2>
  <p class="section-intro">How accurate it is, broken down by what kind of position it's facing. Items marked * are
  proxy categories (estimated from the position itself, not hand-labeled — see glossary).</p>
  {dim_rows}

  <h2>02 — Primary Weakness</h2>
  <div class="weakness-box">{weakness_desc}</div>

  {blunder_section}

  {llm_section}

  <h2>05 — Diagnostic Experiments</h2>
  <p class="section-intro">Each row is a knob we tried turning (e.g. search deeper) to see if it fixed the
  weakness above, measured only on the specific blunder positions used to find it — not yet proof it generalizes.</p>
  <table>
    <tr><th>Hypothesis</th><th>Condition A &rarr; B</th><th>Mean |regret| A &rarr; B</th><th>Improvement</th><th>p-value</th></tr>
    {hyp_rows}
  </table>
  <p class="section-intro" style="margin-top:10px">Ranked by improvement magnitude; the top row is the best-supported
  candidate cause. Small sample sizes mean these numbers should be read as directional, not conclusive — that's what
  section 06 is for.</p>

  <h2>06 — Held-Out Validation: Does the Fix Actually Work?</h2>
  <div class="validation">
    <div class="col before"><div class="n">{_fmt(before)}</div><div>before (baseline)</div></div>
    <div class="arrow">&rarr;</div>
    <div class="col after"><div class="n">{_fmt(after)}</div><div>after (candidate fix)</div></div>
  </div>
  <div class="plain-callout">{intervention_plain}</div>

  <h2>07 — Glossary (for normal people)</h2>
  <dl class="glossary">
    <dt>Decision accuracy</dt>
    <dd>Percent of tested positions where the engine's move was at most 0.5 pawns worse than the best known move — i.e. "close enough to correct."</dd>
    <dt>Regret</dt>
    <dd>How much worse off you are, in pawns of material/position, for playing the engine's move instead of the best known move. 0 = no mistake.</dd>
    <dt>Blunder</dt>
    <dd>A move with high regret — the engine had a much better option available and didn't take it.</dd>
    <dt>Held-out validation</dt>
    <dd>Testing a proposed fix on positions it never saw while the fix was being designed, so we're not just fooling ourselves with a fix that only works on the examples we picked it from.</dd>
    <dt>p-value</dt>
    <dd>A statistical significance measure. Shown as &mdash; wherever we don't have a real one, rather than inventing a number that looks more rigorous than it is.</dd>
    <dt>PV (principal variation)</dt>
    <dd>The sequence of moves the engine expects to follow after its chosen move — its "plan."</dd>
  </dl>

  <footer>
    <strong>Noted deviations from the original spec (no network access in this build environment):</strong>
    <ul>{deviation_items}</ul>
  </footer>

</body>
</html>"""
