"""
GAMBIT chess environment — Layer 1 of the abstract architecture
(see docs/04_abstract_architecture.md).

Self-contained chess rules engine: board representation, legal move
generation, and state transition. No external chess library — this sandbox
had no network access to install python-chess, so full rules (castling, en
passant, promotion, check/checkmate/stalemate) are implemented directly.
This is the Python prototype; port to C++ per docs/05_technical_architecture.md
once the loop above it is proven.
"""

from dataclasses import dataclass
from typing import List, Optional

FILES = "abcdefgh"
RANKS = "12345678"
WHITE, BLACK = "w", "b"

KNIGHT_OFFSETS = [(1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2)]
KING_OFFSETS = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]
BISHOP_DIRS = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
ROOK_DIRS = [(1, 0), (-1, 0), (0, 1), (0, -1)]

START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def sq(file_idx: int, rank_idx: int) -> int:
    return rank_idx * 8 + file_idx


def sq_name(square: int) -> str:
    return FILES[square % 8] + RANKS[square // 8]


def parse_sq(name: str) -> int:
    return sq(FILES.index(name[0]), RANKS.index(name[1]))


@dataclass
class Move:
    from_sq: int
    to_sq: int
    promotion: Optional[str] = None  # 'q','r','b','n'
    is_en_passant: bool = False
    is_castle: bool = False
    castle_side: Optional[str] = None  # 'K','Q','k','q'

    def uci(self) -> str:
        s = sq_name(self.from_sq) + sq_name(self.to_sq)
        return s + self.promotion if self.promotion else s

    def __eq__(self, other):
        return isinstance(other, Move) and self.uci() == other.uci()


class Board:
    """A chess position. Immutable-by-convention: apply_move returns a new Board."""

    def __init__(self, fen: str = START_FEN):
        self.load_fen(fen)

    def load_fen(self, fen: str):
        parts = fen.split()
        board = ["."] * 64
        for r, row in enumerate(parts[0].split("/")):
            rank_idx, file_idx = 7 - r, 0
            for ch in row:
                if ch.isdigit():
                    file_idx += int(ch)
                else:
                    board[sq(file_idx, rank_idx)] = ch
                    file_idx += 1
        self.board = board
        self.turn = parts[1]
        self.castling = "" if parts[2] == "-" else parts[2]
        self.ep_square = None if parts[3] == "-" else parse_sq(parts[3])
        self.halfmove = int(parts[4]) if len(parts) > 4 else 0
        self.fullmove = int(parts[5]) if len(parts) > 5 else 1

    def fen(self) -> str:
        rows = []
        for r in range(7, -1, -1):
            row, empty = "", 0
            for f in range(8):
                p = self.board[sq(f, r)]
                if p == ".":
                    empty += 1
                else:
                    if empty:
                        row += str(empty)
                        empty = 0
                    row += p
            if empty:
                row += str(empty)
            rows.append(row)
        castling = self.castling if self.castling else "-"
        ep = sq_name(self.ep_square) if self.ep_square is not None else "-"
        return f"{'/'.join(rows)} {self.turn} {castling} {ep} {self.halfmove} {self.fullmove}"

    def copy(self) -> "Board":
        b = Board.__new__(Board)
        b.board = self.board[:]
        b.turn, b.castling, b.ep_square = self.turn, self.castling, self.ep_square
        b.halfmove, b.fullmove = self.halfmove, self.fullmove
        return b

    @staticmethod
    def piece_color(piece: str) -> Optional[str]:
        return None if piece == "." else (WHITE if piece.isupper() else BLACK)

    def king_square(self, color: str) -> int:
        return self.board.index("K" if color == WHITE else "k")

    def is_attacked(self, square: int, by_color: str) -> bool:
        f0, r0 = square % 8, square // 8
        pawn_dir = -1 if by_color == WHITE else 1  # square a `by_color` pawn attacking `square` sits at
        pawn = "P" if by_color == WHITE else "p"
        for df in (-1, 1):
            f, r = f0 + df, r0 + pawn_dir
            if 0 <= f < 8 and 0 <= r < 8 and self.board[sq(f, r)] == pawn:
                return True
        knight = "N" if by_color == WHITE else "n"
        for df, dr in KNIGHT_OFFSETS:
            f, r = f0 + df, r0 + dr
            if 0 <= f < 8 and 0 <= r < 8 and self.board[sq(f, r)] == knight:
                return True
        king = "K" if by_color == WHITE else "k"
        for df, dr in KING_OFFSETS:
            f, r = f0 + df, r0 + dr
            if 0 <= f < 8 and 0 <= r < 8 and self.board[sq(f, r)] == king:
                return True
        bishop = "B" if by_color == WHITE else "b"
        rook = "R" if by_color == WHITE else "r"
        queen = "Q" if by_color == WHITE else "q"
        for df, dr in BISHOP_DIRS:
            f, r = f0 + df, r0 + dr
            while 0 <= f < 8 and 0 <= r < 8:
                p = self.board[sq(f, r)]
                if p != ".":
                    if p in (bishop, queen):
                        return True
                    break
                f, r = f + df, r + dr
        for df, dr in ROOK_DIRS:
            f, r = f0 + df, r0 + dr
            while 0 <= f < 8 and 0 <= r < 8:
                p = self.board[sq(f, r)]
                if p != ".":
                    if p in (rook, queen):
                        return True
                    break
                f, r = f + df, r + dr
        return False

    def in_check(self, color: str) -> bool:
        opp = BLACK if color == WHITE else WHITE
        return self.is_attacked(self.king_square(color), opp)

    def pseudo_legal_moves(self) -> List[Move]:
        moves = []
        color, opp = self.turn, (BLACK if self.turn == WHITE else WHITE)
        for square in range(64):
            piece = self.board[square]
            if piece == "." or self.piece_color(piece) != color:
                continue
            f0, r0 = square % 8, square // 8
            kind = piece.upper()
            if kind == "P":
                direction = 1 if color == WHITE else -1
                start_rank = 1 if color == WHITE else 6
                promo_rank = 7 if color == WHITE else 0
                r1 = r0 + direction
                if 0 <= r1 < 8 and self.board[sq(f0, r1)] == ".":
                    if r1 == promo_rank:
                        moves += [Move(square, sq(f0, r1), promotion=p) for p in "qrbn"]
                    else:
                        moves.append(Move(square, sq(f0, r1)))
                        if r0 == start_rank and self.board[sq(f0, r0 + 2 * direction)] == ".":
                            moves.append(Move(square, sq(f0, r0 + 2 * direction)))
                for df in (-1, 1):
                    f1 = f0 + df
                    if 0 <= f1 < 8 and 0 <= r1 < 8:
                        target = sq(f1, r1)
                        tp = self.board[target]
                        if tp != "." and self.piece_color(tp) == opp:
                            if r1 == promo_rank:
                                moves += [Move(square, target, promotion=p) for p in "qrbn"]
                            else:
                                moves.append(Move(square, target))
                        elif self.ep_square is not None and target == self.ep_square:
                            moves.append(Move(square, target, is_en_passant=True))
            elif kind == "N":
                for df, dr in KNIGHT_OFFSETS:
                    f, r = f0 + df, r0 + dr
                    if 0 <= f < 8 and 0 <= r < 8:
                        target = sq(f, r)
                        tp = self.board[target]
                        if tp == "." or self.piece_color(tp) == opp:
                            moves.append(Move(square, target))
            elif kind == "K":
                for df, dr in KING_OFFSETS:
                    f, r = f0 + df, r0 + dr
                    if 0 <= f < 8 and 0 <= r < 8:
                        target = sq(f, r)
                        tp = self.board[target]
                        if tp == "." or self.piece_color(tp) == opp:
                            moves.append(Move(square, target))
                rights = self.castling
                if color == WHITE and square == parse_sq("e1"):
                    if ("K" in rights and self.board[parse_sq("f1")] == "." and self.board[parse_sq("g1")] == "."
                            and not self.in_check(WHITE) and not self.is_attacked(parse_sq("f1"), BLACK)
                            and not self.is_attacked(parse_sq("g1"), BLACK)):
                        moves.append(Move(square, parse_sq("g1"), is_castle=True, castle_side="K"))
                    if ("Q" in rights and self.board[parse_sq("d1")] == "." and self.board[parse_sq("c1")] == "."
                            and self.board[parse_sq("b1")] == "." and not self.in_check(WHITE)
                            and not self.is_attacked(parse_sq("d1"), BLACK) and not self.is_attacked(parse_sq("c1"), BLACK)):
                        moves.append(Move(square, parse_sq("c1"), is_castle=True, castle_side="Q"))
                if color == BLACK and square == parse_sq("e8"):
                    if ("k" in rights and self.board[parse_sq("f8")] == "." and self.board[parse_sq("g8")] == "."
                            and not self.in_check(BLACK) and not self.is_attacked(parse_sq("f8"), WHITE)
                            and not self.is_attacked(parse_sq("g8"), WHITE)):
                        moves.append(Move(square, parse_sq("g8"), is_castle=True, castle_side="k"))
                    if ("q" in rights and self.board[parse_sq("d8")] == "." and self.board[parse_sq("c8")] == "."
                            and self.board[parse_sq("b8")] == "." and not self.in_check(BLACK)
                            and not self.is_attacked(parse_sq("d8"), WHITE) and not self.is_attacked(parse_sq("c8"), WHITE)):
                        moves.append(Move(square, parse_sq("c8"), is_castle=True, castle_side="q"))
            else:
                dirs = BISHOP_DIRS if kind == "B" else ROOK_DIRS if kind == "R" else BISHOP_DIRS + ROOK_DIRS
                for df, dr in dirs:
                    f, r = f0 + df, r0 + dr
                    while 0 <= f < 8 and 0 <= r < 8:
                        target = sq(f, r)
                        tp = self.board[target]
                        if tp == ".":
                            moves.append(Move(square, target))
                        else:
                            if self.piece_color(tp) == opp:
                                moves.append(Move(square, target))
                            break
                        f, r = f + df, r + dr
        return moves

    def apply_move(self, move: Move) -> "Board":
        b = self.copy()
        piece = b.board[move.from_sq]
        color = b.piece_color(piece)
        opp = BLACK if color == WHITE else WHITE
        captured = b.board[move.to_sq]

        if move.is_en_passant:
            cap_sq = move.to_sq + (-8 if color == WHITE else 8)
            b.board[cap_sq] = "."

        b.board[move.to_sq] = piece
        b.board[move.from_sq] = "."

        if move.promotion:
            b.board[move.to_sq] = move.promotion.upper() if color == WHITE else move.promotion.lower()

        if move.is_castle:
            side_map = {
                "K": (parse_sq("f1"), parse_sq("h1")),
                "Q": (parse_sq("d1"), parse_sq("a1")),
                "k": (parse_sq("f8"), parse_sq("h8")),
                "q": (parse_sq("d8"), parse_sq("a8")),
            }
            dest, src = side_map[move.castle_side]
            b.board[dest] = b.board[src]
            b.board[src] = "."

        if piece == "K":
            b.castling = b.castling.replace("K", "").replace("Q", "")
        if piece == "k":
            b.castling = b.castling.replace("k", "").replace("q", "")
        for s, r in [(parse_sq("a1"), "Q"), (parse_sq("h1"), "K"), (parse_sq("a8"), "q"), (parse_sq("h8"), "k")]:
            if move.from_sq == s or move.to_sq == s:
                b.castling = b.castling.replace(r, "")

        if piece.upper() == "P" and abs(move.to_sq - move.from_sq) == 16:
            b.ep_square = (move.from_sq + move.to_sq) // 2
        else:
            b.ep_square = None

        b.halfmove = 0 if (piece.upper() == "P" or captured != "." or move.is_en_passant) else b.halfmove + 1
        if color == BLACK:
            b.fullmove += 1
        b.turn = opp
        return b

    def legal_moves(self) -> List[Move]:
        color = self.turn
        return [m for m in self.pseudo_legal_moves() if not self.apply_move(m).in_check(color)]

    def is_checkmate(self) -> bool:
        return self.in_check(self.turn) and not self.legal_moves()

    def is_stalemate(self) -> bool:
        return not self.in_check(self.turn) and not self.legal_moves()

    def is_terminal(self) -> bool:
        return self.is_checkmate() or self.is_stalemate() or self.halfmove >= 100

    def result(self) -> Optional[str]:
        """'1-0' / '0-1' / '1/2-1/2' / None if the game isn't over."""
        if self.is_checkmate():
            return "0-1" if self.turn == WHITE else "1-0"
        if self.is_stalemate() or self.halfmove >= 100:
            return "1/2-1/2"
        return None

    def game_phase(self) -> str:
        """Rough phase classifier for benchmark tagging (docs/07)."""
        piece_count = sum(1 for p in self.board if p not in (".", "K", "k"))
        if self.fullmove <= 10 and piece_count >= 28:
            return "opening"
        if piece_count <= 12:
            return "endgame"
        return "middlegame"
