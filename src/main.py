from .utils import chess_manager, GameContext
from chess import Move
import random
import time
import os

from .search import evaluate_moves, load_model


# Write code here that runs once
# Can do things like load models from huggingface, make connections to subprocesses, etcwenis

model_path = os.path.join(os.path.dirname(__file__), "value_net_mini.pth")

GLOBAL_MODEL, GLOBAL_DEVICE = load_model(
    model_path=model_path,
    in_channels=18,
    channels=16,
    num_blocks=2,
)


@chess_manager.entrypoint
def test_func(ctx: GameContext):
    # This gets called every time the model needs to make a move
    # Return a python-chess Move object that is a legal move for the current position

    t_total_start = time.perf_counter()


    print("Cooking move...")
    # print(ctx.board.move_stack)
    time.sleep(0.1)

    board = ctx.board

    n_moves = len(list(board.legal_moves))

    max_depth = 2

    print(max_depth)

    t_search_start = time.perf_counter()

    scores = evaluate_moves(
        board=board,
        max_depth=max_depth,            
        model=GLOBAL_MODEL,
        device=GLOBAL_DEVICE,
        verbose=False
    )

    t_search = time.perf_counter() - t_search_start
    print(f"Search time: {t_search:.4f} s")

    t_post_start = time.perf_counter()


    minv = min(scores.values())
    move_weights = {m: (s - minv + 1e-6) for m, s in scores.items()}
    total_weight = sum(move_weights.values())

  
    move_probs = {
        move: weight / total_weight
        for move, weight in move_weights.items()
    }
    ctx.logProbabilities(move_probs)

    
    best_move = max(move_weights.items(), key=lambda x: x[1])[0]

    t_post = time.perf_counter() - t_post_start
    t_total = time.perf_counter() - t_total_start
    print(f"Post-processing time: {t_post:.4f} s")
    print(f"Total test_func time: {t_total:.4f} s")

    print("BEST MOVE:", best_move, "SCORE:", scores[best_move])

    return best_move


@chess_manager.reset
def reset_func(ctx: GameContext):
    # This gets called when a new game begins
    # Should do things like clear caches, reset model state, etc.
    pass
