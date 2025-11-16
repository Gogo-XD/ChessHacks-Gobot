import math
from dataclasses import dataclass
from typing import Optional, Dict

import chess
import torch

from .dataprep import encode_board
from .model import ValueNet

def load_model(
    model_path: str = "/src/value_net.pth",
    in_channels: int = 18,
    channels: int = 32,
    num_blocks: int = 4,
) -> tuple[ValueNet, torch.device]:
    """
    Load the trained value network and return (model, device).
    Adjust channels/num_blocks to match how you trained. 
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ValueNet(in_channels=in_channels, channels=channels, num_blocks=num_blocks).to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, device


def evaluate(board: chess.Board, model: ValueNet, device: torch.device) -> float:
    """
    Evaluate a board position from the POV of the side to move.
    Returns a scalar in roughly [-1, 1].
    """
    if board.is_checkmate():
        return -1.0
    if board.is_stalemate() or board.is_insufficient_material() or board.can_claim_draw():
        return 0.0

    x_np = encode_board(board)
    x = torch.from_numpy(x_np).unsqueeze(0).to(device)  

    with torch.no_grad():
        out = model(x)  
    return float(out.item())


# ---------------- Transposition Table ---------------- #

EXACT, LOWERBOUND, UPPERBOUND = 0, -1, 1


@dataclass
class TTEntry:
    depth: int
    score: float
    flag: int      
    best_move: Optional[chess.Move]


TT: Dict[int, TTEntry] = {}


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
    Return legal moves ordered:
      1) TT move (if any)
      2) Captures, ordered by MVV-LVA
      3) Quiet moves
    """
    moves = list(board.legal_moves)

    def key(move: chess.Move):
        if tt_move is not None and move == tt_move:
            return (3, 0)
        if board.is_capture(move):
            return (2, mvv_lva_score(board, move))
        return (1, 0)

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
    Negamax search with alpha-beta pruning and transposition table.
    Returns value from POV of side to move.
    """
    alpha_orig = alpha
    key = hash(board._transposition_key())

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

    # Leaf / terminal
    if depth == 0 or board.is_game_over():
        return evaluate(board, model, device)

    max_score = -math.inf
    best_move = None

    tt_move = entry.best_move if entry is not None else None

    for move in ordered_moves(board, tt_move):
        board.push(move)
        score = -negamax(board, depth - 1, -beta, -alpha, model, device)
        board.pop()

        if score > max_score:
            max_score = score
            best_move = move
        if score > alpha:
            alpha = score
        if alpha >= beta:
            break  

    # Store in TT
    flag = EXACT
    if max_score <= alpha_orig:
        flag = UPPERBOUND
    elif max_score >= beta:
        flag = LOWERBOUND

    TT[key] = TTEntry(depth=depth, score=max_score, flag=flag, best_move=best_move)
    return max_score


# ---------------- Iterative Deepening ---------------- #

def evaluate_moves(
    board: chess.Board,
    max_depth: int,
    model: ValueNet,
    device: torch.device,
    verbose: bool = False,
):
    """
    Returns a dict {move: score} for all legal moves at the root.
    """
    global TT
    TT = {}  

    legal_moves = list(board.legal_moves)
    scores = {m: 0.0 for m in legal_moves}  

    for depth in range(1, max_depth + 1):
        print("[Depth {}] Iterative deepening...".format(depth))
        root_key = hash(board._transposition_key)
        root_tt = TT.get(root_key)
        tt_move = root_tt.best_move if root_tt is not None else None
        ordered = ordered_moves(board, tt_move)

        if verbose:
            print(f"[Depth {depth}] Evaluating {len(ordered)} moves...")

        for move in ordered:
            board.push(move)
            score = -negamax(board, depth - 1, -math.inf, math.inf, model, device)
            board.pop()

            if depth == max_depth:
                scores[move] = score

    return scores



# # ---------------- Simple CLI Test ---------------- #

# if __name__ == "__main__":
#     # Example usage: search from the starting position
#     model, device = load_model("value_net.pth", in_channels=18, channels=32, num_blocks=4)


#     fen = input("Enter FEN (or leave blank for starting position): ").strip()
#     board = chess.Board(fen if fen else chess.STARTING_FEN)

#     moves = evaluate_moves(board, max_depth=4, model=model, device=device, verbose=True)

#     sorted_moves = sorted(moves.items(), key=lambda x: x[1])

#     # Keep only the top 5 best moves
#     top_5 = sorted_moves[-5:]

#     print("\nTop 5 moves (best last):")
#     for move, score in top_5:
#         print(f"Move: {move}, Score: {score:.4f}")
