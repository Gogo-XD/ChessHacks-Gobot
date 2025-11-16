from .utils import chess_manager, GameContext
from chess import Move
import random
import time
import os

from .search import evaluate_moves, load_model


# Write code here that runs once
# Can do things like load models from huggingface, make connections to subprocesses, etcwenis

model_path = os.path.join(os.path.dirname(__file__), "value_net.pth")

GLOBAL_MODEL, GLOBAL_DEVICE = load_model(
    model_path=model_path,
    in_channels=18,
    channels=32,
    num_blocks=4,
)


@chess_manager.entrypoint
def test_func(ctx: GameContext):
    # This gets called every time the model needs to make a move
    # Return a python-chess Move object that is a legal move for the current position

    print("Cooking move...")
    # print(ctx.board.move_stack)
    time.sleep(0.1)

    board = ctx.board

    scores = evaluate_moves(
        board=board,
        max_depth=2,              # or 5/6 if fast enough
        model=GLOBAL_MODEL,
        device=GLOBAL_DEVICE,
        verbose=True
    )

    print(scores)

    minv = min(scores.values())
    move_weights = {m: (s - minv + 1e-6) for m, s in scores.items()}
    total_weight = sum(move_weights.values())

  
    # Normalize so probabilities sum to 1
    move_probs = {
        move: weight / total_weight
        for move, weight in move_weights.items()
    }
    ctx.logProbabilities(move_probs)

    moves = list(move_probs.keys())
    weights = list(move_probs.values())

    
    move = random.choices(moves, weights=weights, k=1)[0]
    print("MOVE: ", move)

    return move


@chess_manager.reset
def reset_func(ctx: GameContext):
    # This gets called when a new game begins
    # Should do things like clear caches, reset model state, etc.
    pass
