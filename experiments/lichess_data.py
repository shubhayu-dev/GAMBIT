import csv
import io
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import zstandard as zstd
from engine.environment import Board


RAW_ZST_PATH = Path(__file__).resolve().parent.parent / "data" / "lichess_raw" / "lichess_db_puzzle.csv.zst"


def load_lichess_positions(
    path: Path = RAW_ZST_PATH,
    validate: bool = True,
    max_positions: Optional[int] = 500,
    phase_filter: Optional[str] = None,
) -> Tuple[List[Dict], List[Tuple[str, str]]]:
    positions = []
    skipped = []

    def process_stream(reader):
        nonlocal positions, skipped
        for row in reader:
            puzzle_id = row["PuzzleId"]
            moves = row["Moves"].split()
            themes = set(row.get("Themes", "").split())

            if len(moves) < 2:
                skipped.append((puzzle_id, "fewer than 2 moves"))
                continue

            lichess_phase = next((t for t in ("opening", "middlegame", "endgame") if t in themes), None)
            if phase_filter and lichess_phase != phase_filter:
                continue

            try:
                board = Board(row["FEN"])
            except Exception as e:
                skipped.append((puzzle_id, f"FEN failed to parse: {e}"))
                continue

            legal_moves_1 = {m.uci(): m for m in board.legal_moves()}
            if moves[0] not in legal_moves_1:
                if validate:
                    skipped.append((puzzle_id, f"setup move {moves[0]} not legal"))
                    continue
                continue

            puzzle_board = board.apply_move(legal_moves_1[moves[0]])

            legal_moves_2 = {m.uci(): m for m in puzzle_board.legal_moves()}
            if moves[1] not in legal_moves_2:
                if validate:
                    skipped.append((puzzle_id, f"reference move {moves[1]} not legal at puzzle state"))
                    continue
                continue

            # Extract Game ID to maintain split isolation
            game_url = row.get("GameUrl", "")
            game_id = game_url.split("/")[-1].split("#")[0] if game_url else puzzle_id

            positions.append({
                "fen": puzzle_board.fen(),
                "reference_move_lichess": moves[1],
                "solution_moves_lichess": moves[1:],
                "rating": int(row["Rating"]),
                "themes": sorted(themes),
                "lichess_phase": lichess_phase,
                "internal_phase": puzzle_board.game_phase(),
                "source": "lichess",
                "puzzle_id": puzzle_id,
                "game_id": game_id,
            })

            if max_positions and len(positions) >= max_positions:
                break

    if path.suffix == ".zst":
        with open(path, "rb") as fh:
            dctx = zstd.ZstdDecompressor()
            with dctx.stream_reader(fh) as reader:
                text_stream = io.TextIOWrapper(reader, encoding="utf-8")
                process_stream(csv.DictReader(text_stream))
    else:
        with open(path, "r", encoding="utf-8") as fh:
            process_stream(csv.DictReader(fh))

    return positions, skipped

if __name__ == "__main__":
    # Adjust max_positions as needed for your pipeline scale (e.g., 500-2000)
    positions, skipped = load_lichess_positions(max_positions=500, phase_filter="middlegame")
    
    print(f"Loaded {len(positions)} valid positions.")
    print(f"Skipped {len(skipped)} rows.")
    
    if positions:
        output_file = Path(__file__).resolve().parent.parent / "data" / "lichess_puzzles_sample.csv"
        
        with open(output_file, "w", newline="", encoding="utf-8") as f:
            # Dynamically grab the headers from the first dictionary
            writer = csv.DictWriter(f, fieldnames=positions[0].keys())
            writer.writeheader()
            writer.writerows(positions)
            
        print(f"Successfully saved decompressed data to: {output_file}")