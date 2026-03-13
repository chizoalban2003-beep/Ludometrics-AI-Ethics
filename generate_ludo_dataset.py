from __future__ import annotations

import argparse
import csv
import math
import random
from pathlib import Path
from typing import List, Tuple

PLAYER_NAMES = ("Red", "Green", "Yellow", "Blue")
TOKENS_PER_PLAYER = 4
FINISH_POS = 57
BOARD_MAX = 56

COLUMNS = [
    "Game_ID",
    "Turn",
    "Player",
    "Dice_Roll",
    "Token_Moved",
    "Position_Before",
    "Position_After",
    "Tokens_Home",
    "Tokens_Active",
    "Tokens_Finished",
    "Captured_Opponent",
    "Is_Winner",
]


def _next_position(position: int, dice_roll: int) -> int:
    """Compute new token position with bounce-back rule from finish."""
    target = position + dice_roll
    if target > FINISH_POS:
        return FINISH_POS - (target - FINISH_POS)
    return target


def _valid_moves(tokens: List[int], dice_roll: int) -> List[Tuple[int, int, int]]:
    """Return (token_index, before, after) for all valid token moves."""
    moves: List[Tuple[int, int, int]] = []
    for idx, pos in enumerate(tokens):
        if pos == FINISH_POS:
            continue
        if pos == 0:
            if dice_roll == 6:
                moves.append((idx, 0, 1))
            continue
        moves.append((idx, pos, _next_position(pos, dice_roll)))
    return moves


def _choose_move(rng: random.Random, moves: List[Tuple[int, int, int]]) -> Tuple[int, int, int]:
    """Pick a move quickly while preferring advancing the most progress."""
    if len(moves) == 1:
        return moves[0]

    # Favor the move with largest after-position (fast convergence),
    # but keep randomness to diversify the dataset.
    best_after = max(m[2] for m in moves)
    shortlist = [m for m in moves if m[2] == best_after]
    return shortlist[rng.randrange(len(shortlist))]


def _count_token_states(tokens: List[int]) -> Tuple[int, int, int]:
    home = 0
    active = 0
    finished = 0
    for pos in tokens:
        if pos == 0:
            home += 1
        elif pos == FINISH_POS:
            finished += 1
        else:
            active += 1
    return home, active, finished


def _simulate_game(game_id: int, rng: random.Random) -> List[List[object]]:
    """Simulate one full 4-player, 4-token game and return turn-level rows."""
    tokens_by_player = [[0, 0, 0, 0] for _ in range(4)]
    rows: List[List[object]] = []

    turn = 1
    player_idx = 0
    winner_idx = -1

    randint = rng.randint

    while winner_idx == -1:
        dice_roll = randint(1, 6)
        current_tokens = tokens_by_player[player_idx]
        moves = _valid_moves(current_tokens, dice_roll)

        token_moved = 0
        pos_before: object = math.nan
        pos_after: object = math.nan
        captured = 0

        if moves:
            token_idx, before, after = _choose_move(rng, moves)
            token_moved = token_idx + 1
            pos_before = before
            pos_after = after
            current_tokens[token_idx] = after

            if 1 <= after <= BOARD_MAX:
                for opp_idx in range(4):
                    if opp_idx == player_idx:
                        continue
                    opp_tokens = tokens_by_player[opp_idx]
                    for j, opp_pos in enumerate(opp_tokens):
                        if opp_pos == after:
                            opp_tokens[j] = 0
                            captured = 1

            if all(pos == FINISH_POS for pos in current_tokens):
                winner_idx = player_idx

        tokens_home, tokens_active, tokens_finished = _count_token_states(current_tokens)

        rows.append(
            [
                game_id,
                turn,
                PLAYER_NAMES[player_idx],
                dice_roll,
                token_moved,
                pos_before,
                pos_after,
                tokens_home,
                tokens_active,
                tokens_finished,
                captured,
                0,  # set after game ends
            ]
        )

        extra_turn = (dice_roll == 6) or (captured == 1)
        if not extra_turn and winner_idx == -1:
            player_idx = (player_idx + 1) % 4

        turn += 1

    winner_name = PLAYER_NAMES[winner_idx]
    for row in rows:
        row[-1] = 1 if row[2] == winner_name else 0

    return rows


def generate_dataset(min_rows: int, seed: int, raw_output: Path, clean_output: Path | None) -> int:
    """Generate dataset until min_rows is reached. Returns total rows written."""
    rng = random.Random(seed)

    raw_output.parent.mkdir(parents=True, exist_ok=True)
    if clean_output is not None:
        clean_output.parent.mkdir(parents=True, exist_ok=True)

    total_rows = 0
    game_id = 0

    with raw_output.open("w", newline="", encoding="utf-8") as raw_file:
        writer = csv.writer(raw_file)
        writer.writerow(COLUMNS)

        while total_rows < min_rows:
            game_rows = _simulate_game(game_id, rng)
            writer.writerows(game_rows)
            total_rows += len(game_rows)
            game_id += 1

    if clean_output is not None:
        # No extra cleaning needed yet; keep synchronized copy for pipeline compatibility.
        clean_output.write_bytes(raw_output.read_bytes())

    return total_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic Ludo dataset for 4-player, 4-token games.")
    parser.add_argument("--min-rows", type=int, default=11000, help="Minimum number of rows to generate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument(
        "--raw-output",
        type=Path,
        default=Path("data file/Raw_Data/ludo_dataset.csv"),
        help="Path for raw dataset CSV.",
    )
    parser.add_argument(
        "--clean-output",
        type=Path,
        default=Path("data file/Raw_Data/ludo_dataset_cleaned.csv"),
        help="Path for cleaned dataset CSV copy.",
    )

    args = parser.parse_args()

    rows_written = generate_dataset(
        min_rows=max(args.min_rows, 11000),
        seed=args.seed,
        raw_output=args.raw_output,
        clean_output=args.clean_output,
    )

    print(f"Generated {rows_written} rows")
    print(f"Saved raw dataset to: {args.raw_output}")
    print(f"Saved cleaned dataset copy to: {args.clean_output}")


if __name__ == "__main__":
    main()
