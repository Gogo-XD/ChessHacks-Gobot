import math
from dataclasses import dataclass
from typing import Optional, Dict, Tuple, Hashable

import chess
import torch

from .dataprep import encode_board
from .model import ValueNet


def load_model(
    model_path: str = "src/value_net_mini.pth",
    in_channels: int = 18,
    channels: int = 16,
    num_blocks: int = 2,
) -> Tuple[ValueNet, torch.device]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ValueNet(
        in_channels=in_channels,
        channels=channels,
        num_blocks=num_blocks,
    ).to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, device


# ---------------- NN Evaluation + Cache ---------------- #

EVAL_CACHE: Dict[Hashable, float] = {}


def tt_key(board: chess.Board) -> Hashable:
    # python-chess internal Zobrist key is private but stable within a version
    if hasattr(board, "_transposition_key"):
        return board._transposition_key()  # type: ignore[attr-defined]
    return board.fen()


def evaluate(board: chess.Board, model: ValueNet, device: torch.device) -> float:
    # Quick terminal checks
    if board.is_checkmate():
        # Side to move is checkmated
        return -1.0
    if (
        board.is_stalemate()
        or board.is_insufficient_material()
        or board.can_claim_draw()
    ):
        return 0.0

    key = tt_key(board)
    cached = EVAL_CACHE.get(key)
    if cached is not None:
        return cached

    x_np = encode_board(board)
    x = torch.from_numpy(x_np).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(x)
    v = float(out.item())
    EVAL_CACHE[key] = v
    return v


# ---------------- Transposition Table ---------------- #

EXACT, LOWERBOUND, UPPERBOUND = 0, -1, 1


@dataclass
class TTEntry:
    depth: int
    score: float
    flag: int        # EXACT / LOWERBOUND / UPPERBOUND
    best_move: Optional[chess.Move]


TT: Dict[Hashable, TTEntry] = {}

# History heuristic (simple per-move key)
HISTORY: Dict[str, int] = {}


# ---------------- Move Ordering ---------------- #

PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 20000,
}


def mvv_lva_score(board: chess.Board, move: chess.Move) -> int:
    """
    Most Valuable Victim - Least Valuable Attacker heuristic.
    Higher score = more urgent capture.
    Non-captures get score 0.
    """
    if not board.is_capture(move):
        return 0

    victim_sq = move.to_square
    attacker_sq = move.from_square

    victim = board.piece_at(victim_sq)
    attacker = board.piece_at(attacker_sq)

    if victim is None or attacker is None:
        return 0

    v_val = PIECE_VALUES.get(victim.piece_type, 0)
    a_val = PIECE_VALUES.get(attacker.piece_type, 0)
    return v_val * 10 - a_val


def ordered_moves(board: chess.Board, tt_move: Optional[chess.Move] = None):
    """
    Order moves using:
      - TT move first
      - captures by MVV-LVA
      - history heuristic for quiet moves
    """
    moves = list(board.legal_moves)

    def key(move: chess.Move):
        score = 0

        if tt_move is not None and move == tt_move:
            score += 1_000_000

        if board.is_capture(move):
            score += 100_000 + mvv_lva_score(board, move)
        else:
            # Quiet move: use history heuristic
            mkey = move.uci()
            score += HISTORY.get(mkey, 0)

        return score

    moves.sort(key=key, reverse=True)
    return moves


# ---------------- Negamax + Alpha-Beta ---------------- #

def negamax(
    board: chess.Board,
    depth: int,
    alpha: float,
    beta: float,
    model: ValueNet,
    device: torch.device,
) -> float:
    """
    Negamax search with alpha-beta pruning, transposition table,
    null-move pruning, simple futility pruning, and late-move reductions.
    Returns value from POV of side to move.
    """
    alpha_orig = alpha
    key = tt_key(board)

    # Transposition table lookup
    entry = TT.get(key)
    if entry is not None and entry.depth >= depth:
        if entry.flag == EXACT:
            return entry.score
        elif entry.flag == LOWERBOUND:
            alpha = max(alpha, entry.score)
        elif entry.flag == UPPERBOUND:
            beta = min(beta, entry.score)
        if alpha >= beta:
            return entry.score

    # Terminal nodes: game over -> direct eval
    if board.is_game_over():
        return evaluate(board, model, device)

    # Leaf: direct NN eval (no quiescence)
    if depth == 0:
        return evaluate(board, model, device)

    in_check = board.is_check()

    # --- Null-move pruning ---
    # Skip null-move if in check or in shallow positions
    if depth >= 3 and not in_check:
        board.push(chess.Move.null())
        # Reduce depth more aggressively here for speed
        score = -negamax(board, depth - 3, -beta, -beta + 1, model, device)
        board.pop()
        if score >= beta:
            return score
    # --------------------------

    # --- Futility pruning at shallow depth (no checks) ---
    if depth == 1 and not in_check:
        # If there are no captures, we can approximate with static eval.
        if not any(board.is_capture(m) for m in board.legal_moves):
            static_eval = evaluate(board, model, device)
            FUT_MARGIN = 0.20  # slightly larger margin for more pruning
            if static_eval + FUT_MARGIN <= alpha:
                return static_eval
    # -----------------------------------------------------

    max_score = -math.inf
    best_move: Optional[chess.Move] = None

    tt_move = entry.best_move if entry is not None else None
    moves = ordered_moves(board, tt_move)

    for idx, move in enumerate(moves):
        is_capture = board.is_capture(move)

        # Late-move reductions for non-captures, later moves, and non-check
        reduction = 0
        if (
            depth >= 3
            and idx >= 4
            and not is_capture
            and not in_check
        ):
            reduction = 1

        search_depth = depth - 1 - reduction

        board.push(move)
        score = -negamax(board, search_depth, -beta, -alpha, model, device)
        board.pop()

        # If we reduced and it looks promising, re-search at full depth
        if reduction > 0 and score > alpha:
            board.push(move)
            score = -negamax(board, depth - 1, -beta, -alpha, model, device)
            board.pop()

        if score > max_score:
            max_score = score
            best_move = move

        if score > alpha:
            alpha = score

        if alpha >= beta:
            # Update history for quiet moves that cause cutoffs
            if not is_capture:
                mkey = move.uci()
                HISTORY[mkey] = HISTORY.get(mkey, 0) + depth * depth
            break

    # Store in TT
    flag = EXACT
    if max_score <= alpha_orig:
        flag = UPPERBOUND
    elif max_score >= beta:
        flag = LOWERBOUND

    TT[key] = TTEntry(depth=depth, score=max_score, flag=flag, best_move=best_move)
    return max_score


# ---------------- Root Search (no Iterative Deepening) ---------------- #

def evaluate_moves(
    board: chess.Board,
    max_depth: int,
    model: ValueNet,
    device: torch.device,
    verbose: bool = False,
):
    """
    Returns a dict {move: score} for all legal moves at the root.
    Single search at max_depth (no iterative deepening) for speed.
    """
    global TT, EVAL_CACHE, HISTORY
    TT = {}
    EVAL_CACHE = {}
    HISTORY = {}

    legal_moves = list(board.legal_moves)
    scores: Dict[chess.Move, float] = {}

    if verbose:
        print(f"[Root] Searching depth {max_depth} over {len(legal_moves)} moves...")

    # Root move ordering (TT is empty at first call)
    ordered = ordered_moves(board, None)

    for move in ordered:
        board.push(move)
        score = -negamax(board, max_depth - 1, -math.inf, math.inf, model, device)
        board.pop()
        scores[move] = score

    return scores

# ---------------- Simple CLI Test (optional) ---------------- #
# if __name__ == "__main__":
#     model, device = load_model("src/value_net.pth", in_channels=18, channels=32, num_blocks=4)
#     fen = input("Enter FEN (or leave blank for starting position): ").strip()
#     board = chess.Board(fen if fen else chess.STARTING_FEN)
#     moves = evaluate_moves(board, max_depth=4, model=model, device=device, verbose=True)
#     sorted_moves = sorted(moves.items(), key=lambda x: x[1])
#     top_5 = sorted_moves[-5:]
#     print("\nTop 5 moves (best last):")
#     for move, score in top_5:
#         print(f"Move: {move}, Score: {score:.4f}")
