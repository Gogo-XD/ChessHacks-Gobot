import chess
import numpy as np

PIECE_PLANES = {
    chess.PAWN:   0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK:   3,
    chess.QUEEN:  4,
    chess.KING:   5,
}

def encode_board(board: chess.Board) -> np.ndarray:
    """
    Encode a chess.Board into a (18, 8, 8) float32 tensor.
    0-5: white pieces (P, N, B, R, Q, K)
    6-11: black pieces (p, n, b, r, q, k)
    12: side-to-move
    13-16: castling rights (WK, WQ, BK, BQ)
    17: en-passant file
    """
    planes = np.zeros((18, 8, 8), dtype=np.float32)

    # Pieces
    for square, piece in board.piece_map().items():
        piece_type = piece.piece_type
        color = piece.color  
        rank = chess.square_rank(square)  
        file = chess.square_file(square)  
        base = 0 if color else 6
        planes[base + PIECE_PLANES[piece_type], rank, file] = 1.0

    if board.turn == chess.WHITE:
        planes[12, :, :] = 1.0
    else:
        planes[12, :, :] = 0.0

    if board.has_kingside_castling_rights(chess.WHITE):
        planes[13, :, :] = 1.0
    if board.has_queenside_castling_rights(chess.WHITE):
        planes[14, :, :] = 1.0
    if board.has_kingside_castling_rights(chess.BLACK):
        planes[15, :, :] = 1.0
    if board.has_queenside_castling_rights(chess.BLACK):
        planes[16, :, :] = 1.0

    if board.ep_square is not None:
        ep_file = chess.square_file(board.ep_square)
        planes[17, :, ep_file] = 1.0

    return planes


def eval_to_target(cp, mate, decisive_cp=800.0) -> float:
    """
    Convert (cp, mate) to a scalar in [-1, 1] from POV of side to move.
    - If mate is not None: ±1
    - Else: cp normalized and clipped.
    """
    if mate is not None:
        return 1.0 if mate > 0 else -1.0

    if cp is None:
        return 0.0

    value = float(cp) / decisive_cp
    if value > 1.0:
        value = 1.0
    elif value < -1.0:
        value = -1.0
    return value
