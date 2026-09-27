"""
Human-readable move commentary report (companion to dashboard/report_html.py's
diagnosis report, same visual language -- dark/gold theme, same CSS variable
names -- so a person switches between the two reports without the aesthetic
shifting under them).

Consumes the OUTPUT of run_commentary_demo.py's pipeline: the rule-based
results (move_commentary_rules.classify_game()) merged with the LLM
commentary (commentary.annotate_game()). Renders one browser-openable HTML
file with a card per move: the position before the move (as an actual
little board, not a raw FEN string), the move, a colored tag, and the
explanation -- plus, for LLM-sourced moves, the search-coverage context
(how many legal moves were actually compared) so the reader can see when an
explanation is grounded in a thin comparison versus a fuller one.
"""

from typing import Dict, List, Optional

TAG_COLORS = {
    "blunder": "var(--bad)",
    "tactical": "var(--ai)",
    "positional": "var(--gold)",
    "book": "var(--ink-dim)",
    "recapture": "var(--ink-dim)",
    "only_legal": "var(--ink-dim)",
    "forced": "var(--ink-dim)",
    "mate": "var(--bad)",
}

UNICODE_PIECES = {
    "P": "&#9817;", "N": "&#9816;", "B": "&#9815;", "R": "&#9814;", "Q": "&#9813;", "K": "&#9812;",
    "p": "&#9823;", "n": "&#9822;", "b": "&#9821;", "r": "&#9820;", "q": "&#9819;", "k": "&#9818;",
}

FILES = "abcdefgh"


def _square_to_rc(square: str) -> Optional[tuple]:
    """'e4' -> (row, col) in an 8x8 grid, row 0 = rank 8 (top), matching how
    the board is rendered below. Returns None for anything malformed."""
    if not square or len(square) < 2 or square[0] not in FILES:
        return None
    try:
        col = FILES.index(square[0])
        row = 8 - int(square[1])
        return row, col
    except (ValueError, IndexError):
        return None


def _render_board_html(fen: str, move_uci: str = "") -> str:
    """Renders the position (before the move) as an 8x8 unicode-piece grid,
    highlighting the from/to squares of the move about to be played so the
    reader can see what's about to happen without cross-referencing UCI
    notation by hand."""
    piece_placement = fen.split(" ")[0]
    rows = piece_placement.split("/")

    from_sq = _square_to_rc(move_uci[0:2]) if len(move_uci) >= 4 else None
    to_sq = _square_to_rc(move_uci[2:4]) if len(move_uci) >= 4 else None

    cells = []
    for r, row in enumerate(rows):
        col = 0
        for ch in row:
            if ch.isdigit():
                for _ in range(int(ch)):
                    cells.append((r, col, None))
                    col += 1
            else:
                cells.append((r, col, ch))
                col += 1

    squares_html = []
    for r, c, piece in cells:
        light = (r + c) % 2 == 0
        classes = ["sq", "light" if light else "dark"]
        if (r, c) == from_sq:
            classes.append("from-sq")
        if (r, c) == to_sq:
            classes.append("to-sq")
        glyph = UNICODE_PIECES.get(piece, "") if piece else ""
        squares_html.append(f'<div class="{" ".join(classes)}">{glyph}</div>')

    return f'<div class="board">{"".join(squares_html)}</div>'


def _move_card(entry: Dict) -> str:
    """One card. `entry` is a normalized dict -- see merge_commentary()."""
    tag = entry.get("tag") or "book"
    color = TAG_COLORS.get(tag, "var(--ink-dim)")
    coverage_html = ""
    if entry.get("n_legal_at_root") is not None and entry.get("n_compared") is not None:
        coverage_html = (
            f'<div class="coverage">compared {entry["n_compared"]} of '
            f'{entry["n_legal_at_root"]} legal moves before choosing</div>'
        )
    board_html = _render_board_html(entry["fen_before"], entry["move"]) if entry.get("fen_before") else ""
    eval_html = ""
    if entry.get("eval_before") is not None and entry.get("eval_after") is not None:
        eval_html = f'<div class="evalline">{entry["eval_before"]:+.2f} &rarr; {entry["eval_after"]:+.2f}</div>'

    return f"""
    <div class="move-card">
      <div class="move-card-board">{board_html}</div>
      <div class="move-card-body">
        <div class="move-card-head">
          <span class="ply">ply {entry['ply']}</span>
          <span class="move mono">{entry['move']}</span>
          <span class="tag" style="background:{color}">{tag}</span>
        </div>
        {eval_html}
        <div class="explanation">{entry['explanation']}</div>
        {coverage_html}
      </div>
    </div>"""


def merge_commentary(rule_based_results: List, llm_commentary: List[Dict], trajectory: List[Dict]) -> List[Dict]:
    """Combines move_commentary_rules.classify_game()'s results and
    commentary.annotate_game()'s LLM output into one ply-ordered list of
    plain dicts the report renderer above understands. `trajectory` (the
    same list run_commentary_demo.py builds) supplies fen_before/eval data
    that neither results list carries on its own.

    Note: this function only ARRANGES already-produced results into display
    order -- it makes no judgment calls Claude authored, so nothing here is
    persisted or altered from what the rule-based pass / LLM already said.
    """
    trajectory_by_ply = {step["ply"]: step for step in trajectory}
    merged: Dict[int, Dict] = {}

    for r in rule_based_results:
        step = trajectory_by_ply.get(r.ply, {})
        merged[r.ply] = {
            "ply": r.ply,
            "move": r.move_uci,
            "tag": r.classification,
            "explanation": r.explanation,
            "fen_before": step.get("fen_before"),
            "eval_before": step.get("eval_before"),
            "eval_after": step.get("eval_after"),
            "n_legal_at_root": None,
            "n_compared": None,
            "source": "rule",
        }

    for c in llm_commentary:
        step = trajectory_by_ply.get(c["ply"], {})
        merged[c["ply"]] = {
            "ply": c["ply"],
            "move": c["move"],
            "tag": c.get("tag"),
            "explanation": c["explanation"],
            "fen_before": step.get("fen_before"),
            "eval_before": step.get("eval_before"),
            "eval_after": step.get("eval_after"),
            "n_legal_at_root": step.get("n_legal_at_root"),
            "n_compared": len(step.get("root_trace", [])) or None,
            "source": "llm",
        }

    return [merged[ply] for ply in sorted(merged.keys())]


def build_commentary_report_html(game_title: str, moves: List[Dict]) -> str:
    """`moves` is merge_commentary()'s output. Returns a complete,
    self-contained HTML document -- same dark/gold theme as
    dashboard/report_html.py's diagnosis report."""
    cards_html = "".join(_move_card(m) for m in moves)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>GAMBIT Move Commentary — {game_title}</title>
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
    max-width: 820px;
    margin: 0 auto;
    padding: 48px 24px 80px;
    line-height: 1.5;
  }}
  .mono {{ font-family: 'SF Mono', 'JetBrains Mono', Menlo, monospace; }}
  h1 {{ font-size: 28px; margin: 0 0 4px; letter-spacing: 0.2px; }}
  .subtitle {{ color: var(--ink-dim); font-size: 14px; margin-bottom: 40px; }}
  .move-card {{
    display: flex;
    gap: 20px;
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: 3px;
    padding: 18px;
    margin-bottom: 16px;
  }}
  .move-card-board {{ flex-shrink: 0; }}
  .board {{
    display: grid;
    grid-template-columns: repeat(8, 22px);
    grid-template-rows: repeat(8, 22px);
    border: 1px solid var(--line);
  }}
  .sq {{ display: flex; align-items: center; justify-content: center; font-size: 16px; }}
  .sq.light {{ background: #2a251c; }}
  .sq.dark {{ background: #17140f; }}
  .sq.from-sq {{ box-shadow: inset 0 0 0 2px var(--bad); }}
  .sq.to-sq {{ box-shadow: inset 0 0 0 2px var(--good); }}
  .move-card-body {{ flex: 1; min-width: 0; }}
  .move-card-head {{ display: flex; align-items: center; gap: 10px; margin-bottom: 6px; }}
  .ply {{ color: var(--ink-dim); font-size: 12px; font-family: monospace; }}
  .move {{ font-size: 16px; color: var(--gold); }}
  .tag {{
    font-size: 11px;
    padding: 2px 8px;
    border-radius: 10px;
    color: #17140f;
    font-family: monospace;
    text-transform: uppercase;
    letter-spacing: 0.3px;
  }}
  .evalline {{ font-family: monospace; font-size: 12px; color: var(--ink-dim); margin-bottom: 8px; }}
  .explanation {{ font-size: 14px; }}
  .coverage {{ font-size: 11px; color: var(--ink-dim); margin-top: 8px; font-family: monospace; }}
  footer {{ margin-top: 40px; font-size: 12px; color: var(--ink-dim); border-top: 1px solid var(--line); padding-top: 16px; }}
</style>
</head>
<body>

  <h1>GAMBIT Move Commentary</h1>
  <div class="subtitle">{game_title} &mdash; {len(moves)} moves explained</div>

  {cards_html}

  <footer>
    Red outline = from-square, green outline = to-square. "Compared N of M legal moves"
    reflects the search's own time-bounded comparison, captured live -- not a
    reconstruction with unlimited time (see engine/search.py's root_trace).
  </footer>

</body>
</html>"""