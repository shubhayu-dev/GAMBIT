"""
Self-contained FEN -> SVG chess board renderer for the dashboard.

No python-chess / cairosvg / external dependency and no browser JS: this
produces a plain inline <svg> string so the dashboard stays a single
self-contained HTML file. Used to visually show "what the board looked
like" for a blunder case study, with arrows for the move the agent played
vs. the reference (correct) move.

Deliberately NOT reusing engine/environment.py's Board class here -- the
dashboard should be able to render a position from saved JSONL data alone,
without importing the engine package or its sys.path tricks.
"""

from typing import Optional

FILES = "abcdefgh"

# Hand-drawn vector pieces (simplified, flat-icon style), each in its own
# local 0-100 x 0-100 box, bottom-weighted so they sit naturally on a
# square. Deliberately NOT font-glyph-based: Unicode chess symbols
# (U+2654-265F) depend on a symbol font being installed and selected by
# the renderer, which is inconsistent across browsers/viewers/PDF export
# and rendered as blank "tofu" boxes in this sandbox's own renderer even
# with Noto Sans Symbols installed -- vector shapes always render
# identically everywhere with zero font dependency.
_PIECE_PATHS = {
    "P": [  # pawn: head + flared body + base
        '<circle cx="50" cy="32" r="13"/>',
        '<path d="M37,44 Q50,39 63,44 L71,76 Q50,82 29,76 Z"/>',
        '<rect x="21" y="76" width="58" height="10" rx="3"/>',
    ],
    "R": [  # rook: crenellated top + body + base
        '<path d="M27,17 L27,31 L35,31 L35,23 L43,23 L43,31 L57,31 L57,23 L65,23 L65,31 L73,31 L73,17 '
        'L65,17 L65,21 L57,21 L57,17 L43,17 L43,21 L35,21 L35,17 Z"/>',
        '<path d="M31,31 L69,31 L63,70 L37,70 Z"/>',
        '<rect x="21" y="70" width="58" height="12" rx="2"/>',
    ],
    "N": [  # knight: stylised horse head + neck + base
        '<path d="M28,80 Q23,80 23,71 Q23,54 39,44 Q33,36 41,26 Q50,15 64,18 Q73,20 71,28 '
        'Q66,25 60,29 Q69,34 72,45 Q75,56 68,64 L73,80 Z"/>',
        '<circle cx="58" cy="29" r="2.6" fill-opacity="0.9"/>',
        '<rect x="20" y="80" width="60" height="8" rx="2"/>',
    ],
    "B": [  # bishop: ball + mitre head w/ slit + body + base
        '<circle cx="50" cy="13" r="4"/>',
        '<circle cx="50" cy="29" r="11"/>',
        '<path d="M37,46 Q50,39 63,46 L69,76 Q50,83 31,76 Z"/>',
        '<rect x="23" y="76" width="54" height="10" rx="3"/>',
    ],
    "Q": [  # queen: crown of 5 balls + body + base
        '<circle cx="24" cy="31" r="6.5"/>', '<circle cx="37.5" cy="21" r="6.5"/>',
        '<circle cx="50" cy="16" r="6.5"/>', '<circle cx="62.5" cy="21" r="6.5"/>',
        '<circle cx="76" cy="31" r="6.5"/>',
        '<path d="M29,37 Q50,31 71,37 L75,76 Q50,83 25,76 Z"/>',
        '<rect x="19" y="76" width="62" height="10" rx="3"/>',
    ],
    "K": [  # king: cross + collar + body + base
        '<rect x="47" y="6" width="6" height="17" rx="1.5"/>',
        '<rect x="39.5" y="12" width="21" height="6" rx="1.5"/>',
        '<rect x="35" y="27" width="30" height="11" rx="3"/>',
        '<path d="M33,39 Q50,33 67,39 L73,76 Q50,83 27,76 Z"/>',
        '<rect x="21" y="76" width="58" height="10" rx="3"/>',
    ],
}


def _piece_svg_group(piece_char: str, x: float, y: float, sq_size: float) -> str:
    """Places one piece (by FEN letter) inside the square whose top-left
    corner is (x, y) and whose side length is sq_size."""
    kind = piece_char.upper()
    paths = _PIECE_PATHS.get(kind)
    if not paths:
        return ""
    is_white = piece_char.isupper()
    fill = "#f5efe0" if is_white else "#241d14"
    stroke = "#241d14" if is_white else "#f5efe0"
    stroke_width = 2.2 if is_white else 0
    scale = (sq_size * 0.82) / 100.0
    offset = (sq_size - 100 * scale) / 2
    body = "".join(paths)
    return (
        f'<g transform="translate({x + offset:.2f},{y + offset - sq_size * 0.04:.2f}) scale({scale:.4f})" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width}" stroke-linejoin="round">{body}</g>'
    )


LIGHT_SQ = "#e9d9b8"
DARK_SQ = "#8a6240"
HIGHLIGHT_FROM = "#f3d250"   # square the moving piece left
HIGHLIGHT_TO_CHOSEN = "#c0564a"    # square the agent's move landed on
HIGHLIGHT_TO_REF = "#6fae5e"       # square the reference move lands on
ARROW_CHOSEN = "#c0564a"
ARROW_REF = "#6fae5e"


def _parse_fen_board(fen: str):
    """Returns an 8x8 dict {(file_idx 0-7, rank_idx 0-7): piece_char}."""
    board = {}
    placement = fen.split()[0]
    rows = placement.split("/")
    for r, row in enumerate(rows):
        rank_idx = 7 - r  # FEN rank 8 first
        file_idx = 0
        for ch in row:
            if ch.isdigit():
                file_idx += int(ch)
            else:
                board[(file_idx, rank_idx)] = ch
                file_idx += 1
    return board


def _sq_center(file_idx: int, rank_idx: int, sq_size: int, margin: int):
    x = margin + file_idx * sq_size + sq_size / 2
    y = margin + (7 - rank_idx) * sq_size + sq_size / 2
    return x, y


def _parse_uci(uci: Optional[str]):
    if not uci or len(uci) < 4:
        return None, None
    from_sq = uci[0:2]
    to_sq = uci[2:4]
    f0, r0 = FILES.index(from_sq[0]), int(from_sq[1]) - 1
    f1, r1 = FILES.index(to_sq[0]), int(to_sq[1]) - 1
    return (f0, r0), (f1, r1)


def render_board_svg(
    fen: str,
    chosen_uci: Optional[str] = None,
    reference_uci: Optional[str] = None,
    size: int = 320,
    caption_chosen: str = "Agent played",
    caption_ref: str = "Correct move",
) -> str:
    """Renders one board position as an inline SVG string, from White's
    point of view (a1 bottom-left). Draws:
      - a gold highlight on the from/to squares of the agent's move
      - a red arrow for the agent's move
      - a green arrow for the reference move (only drawn if it differs
        from the agent's move -- if they match, only one green arrow shows)
    """
    sq_size = size // 8
    margin = 22
    total = size + margin * 2
    board = _parse_fen_board(fen)

    chosen_from, chosen_to = _parse_uci(chosen_uci)
    ref_from, ref_to = _parse_uci(reference_uci)
    same_move = chosen_uci is not None and chosen_uci == reference_uci

    parts = [f'<svg viewBox="0 0 {total} {total}" xmlns="http://www.w3.org/2000/svg" '
             f'font-family="Georgia, serif" class="chessboard-svg">']
    parts.append(f'<rect x="0" y="0" width="{total}" height="{total}" fill="none"/>')

    # squares
    for file_idx in range(8):
        for rank_idx in range(8):
            x = margin + file_idx * sq_size
            y = margin + (7 - rank_idx) * sq_size
            is_light = (file_idx + rank_idx) % 2 == 1
            color = LIGHT_SQ if is_light else DARK_SQ
            parts.append(f'<rect x="{x}" y="{y}" width="{sq_size}" height="{sq_size}" fill="{color}"/>')

    # square highlights (drawn after base squares, before pieces)
    def highlight(cell, color, opacity=0.65):
        if cell is None:
            return
        f, r = cell
        x = margin + f * sq_size
        y = margin + (7 - r) * sq_size
        parts.append(
            f'<rect x="{x}" y="{y}" width="{sq_size}" height="{sq_size}" '
            f'fill="{color}" fill-opacity="{opacity}"/>'
        )

    if not same_move:
        highlight(chosen_from, HIGHLIGHT_FROM, 0.55)
        highlight(chosen_to, HIGHLIGHT_TO_CHOSEN, 0.55)
        highlight(ref_to, HIGHLIGHT_TO_REF, 0.45)
    else:
        highlight(chosen_from, HIGHLIGHT_FROM, 0.55)
        highlight(chosen_to, HIGHLIGHT_TO_REF, 0.5)

    # file/rank labels
    for file_idx in range(8):
        x = margin + file_idx * sq_size + sq_size / 2
        parts.append(
            f'<text x="{x}" y="{margin + 8 * sq_size + 15}" font-size="11" '
            f'fill="#a89b83" text-anchor="middle">{FILES[file_idx]}</text>'
        )
    for rank_idx in range(8):
        y = margin + (7 - rank_idx) * sq_size + sq_size / 2 + 4
        parts.append(
            f'<text x="{margin - 10}" y="{y}" font-size="11" '
            f'fill="#a89b83" text-anchor="middle">{rank_idx + 1}</text>'
        )

    # pieces (hand-drawn vector shapes, see _piece_svg_group -- no font dependency)
    for (file_idx, rank_idx), piece in board.items():
        x = margin + file_idx * sq_size
        y = margin + (7 - rank_idx) * sq_size
        parts.append(_piece_svg_group(piece, x, y, sq_size))

    # arrow marker defs
    parts.append('<defs>'
                  f'<marker id="arrowhead-chosen" markerWidth="8" markerHeight="8" refX="5" refY="4" orient="auto">'
                  f'<polygon points="0 0, 8 4, 0 8" fill="{ARROW_CHOSEN}"/></marker>'
                  f'<marker id="arrowhead-ref" markerWidth="8" markerHeight="8" refX="5" refY="4" orient="auto">'
                  f'<polygon points="0 0, 8 4, 0 8" fill="{ARROW_REF}"/></marker>'
                  '</defs>')

    def arrow(from_cell, to_cell, color, marker_id, offset=0):
        if from_cell is None or to_cell is None or from_cell == to_cell:
            return
        x1, y1 = _sq_center(from_cell[0], from_cell[1], sq_size, margin)
        x2, y2 = _sq_center(to_cell[0], to_cell[1], sq_size, margin)
        # pull the arrow tip back a bit so it doesn't hide under the piece glyph
        dx, dy = x2 - x1, y2 - y1
        dist = max((dx ** 2 + dy ** 2) ** 0.5, 1e-6)
        pull = sq_size * 0.32
        x2 -= dx / dist * pull
        y2 -= dy / dist * pull
        # small perpendicular offset so overlapping arrows are both visible
        px, py = -dy / dist * offset, dx / dist * offset
        parts.append(
            f'<line x1="{x1+px:.1f}" y1="{y1+py:.1f}" x2="{x2+px:.1f}" y2="{y2+py:.1f}" '
            f'stroke="{color}" stroke-width="5" stroke-linecap="round" '
            f'marker-end="url(#{marker_id})" opacity="0.9"/>'
        )

    if same_move:
        arrow(chosen_from, chosen_to, ARROW_REF, "arrowhead-ref")
    else:
        arrow(ref_from, ref_to, ARROW_REF, "arrowhead-ref", offset=4)
        arrow(chosen_from, chosen_to, ARROW_CHOSEN, "arrowhead-chosen", offset=-4)

    parts.append("</svg>")
    return "".join(parts)
