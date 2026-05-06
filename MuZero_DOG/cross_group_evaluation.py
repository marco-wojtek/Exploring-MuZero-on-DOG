"""
Cross-Group Evaluation: Belief-State agents vs Non-Belief-State agents.

Every agent from Group A (trained WITH belief states) plays against
every agent from Group B (trained WITHOUT belief states).

No baseline tests, no intra-group matchups, no symmetric reversal.
"""
import sys, os

import wandb
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
import jax
import numpy as np
from time import time

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
from MuZero_DOG.muzero_dog import load_params_from_file
from MuZero_DOG.eval_config.eval_cross_group import evaluate_cross_group

FOLDER = "MuZero_DOG/models/params/"
BATCH_SIZE = 250  # parallel games per matchup (total = BATCH_SIZE * 4 per matchup)

# ---------------------------------------------------------------------------
# Group A: trained WITH belief states (encode_board_with_belief)
# ---------------------------------------------------------------------------
GROUP_A_AGENTS = [
    f"{FOLDER}muzero_dog_params_lr0.001_g500_it100_seed25.pkl",  # session
    f"{FOLDER}muzero_dog_params_lr0.001_g500_it100_seed27.pkl",  # session
    f"{FOLDER}muzero_dog_params_lr0.001_g500_it100_seed28.pkl",  # classic
    f"{FOLDER}muzero_dog_params_lr0.001_g500_it100_seed36.pkl",  # classic
    f"{FOLDER}muzero_dog_params_lr0.001_g500_it100_seed33.pkl",  # sample
    f"{FOLDER}muzero_dog_params_lr0.001_g500_it100_seed34.pkl",  # sample
]
GROUP_A_TYPES = [4, 4, 2, 2, 3, 3]
GROUP_A_BELIEF = True  # all use encode_board_with_belief

# ---------------------------------------------------------------------------
# Group B: trained WITHOUT belief states (encode_board)
# ---------------------------------------------------------------------------
GROUP_B_AGENTS = [
    f"{FOLDER}muzero_dog_params_lr0.001_g500_it100_seed23.pkl",  # session
    f"{FOLDER}muzero_dog_params_lr0.001_g500_it100_seed24.pkl",  # session
    f"{FOLDER}muzero_dog_params_lr0.001_g500_it100_seed40.pkl",  # classic
    f"{FOLDER}muzero_dog_params_lr0.001_g500_it100_seed41.pkl",  # classic
    f"{FOLDER}muzero_dog_params_lr0.001_g500_it100_seed22.pkl",  # sample
    f"{FOLDER}muzero_dog_params_lr0.001_g500_it100_seed39.pkl",  # sample
]
GROUP_B_TYPES = [4, 4, 2, 2, 3, 3]
GROUP_B_BELIEF = False  # all use encode_board


def short_name(path):
    base = os.path.splitext(os.path.basename(path))[0]
    # extract seed number for brevity
    parts = base.split('_')
    seed = [p for p in parts if p.startswith('seed')]
    return seed[0] if seed else base


def run_cross_group_evaluation(batch_size=BATCH_SIZE):
    """
    Paired evaluation: agent i from Group A vs agent i from Group B.
    Group A is always Team A (players 0 & 2), Group B is Team B (players 1 & 3).
    """
    assert len(GROUP_A_AGENTS) == len(GROUP_B_AGENTS), "Groups must have equal length"
    total_matchups = len(GROUP_A_AGENTS)
    print(f"\n{'=' * 70}")
    print(f"CROSS-GROUP EVALUATION: Belief vs Non-Belief (paired)")
    print(f"Group A ({len(GROUP_A_AGENTS)} agents, belief=True) vs "
          f"Group B ({len(GROUP_B_AGENTS)} agents, belief=False)")
    print(f"Total matchups: {total_matchups}")
    print(f"Games per matchup: {batch_size * 4}")
    print(f"{'=' * 70}")

    results = {}
    total_start = time()

    for i, (spec_a, type_a, spec_b, type_b) in enumerate(
        zip(GROUP_A_AGENTS, GROUP_A_TYPES, GROUP_B_AGENTS, GROUP_B_TYPES)
    ):
        params_a = load_params_from_file(spec_a)
        params_b = load_params_from_file(spec_b)
        name_a = short_name(spec_a)
        name_b = short_name(spec_b)

        print(f"\n[{i + 1}/{total_matchups}]  "
              f"A: {name_a} (type={type_a}, belief)  vs  "
              f"B: {name_b} (type={type_b}, no-belief)")

        t0 = time()
        result = evaluate_cross_group(
            params_a, params_b,
            type_a=type_a, type_b=type_b,
            belief_a=GROUP_A_BELIEF, belief_b=GROUP_B_BELIEF,
            batch_size=batch_size,
        )
        elapsed = time() - t0
        print(f"  done in {elapsed:.1f}s")

        results[(name_a, name_b)] = result

    total_elapsed = time() - total_start
    print(f"\n{'=' * 70}")
    print(f"Cross-group evaluation complete in {total_elapsed:.1f}s  "
          f"({total_matchups} matchups)")
    print(f"{'=' * 70}")

    # Summary table
    print(f"\n{'=' * 70}")
    print("SUMMARY: Team A (belief) wins / Team B (no-belief) wins")
    print(f"{'=' * 70}")
    for i in range(total_matchups):
        name_a = short_name(GROUP_A_AGENTS[i])
        name_b = short_name(GROUP_B_AGENTS[i])
        r = results[(name_a, name_b)]
        print(f"  {name_a} vs {name_b}:  {r['team_a_wins']:>3} / {r['team_b_wins']:<3}")

    return results


if __name__ == "__main__":
    # Check existence of all agent files before starting
    for spec in GROUP_A_AGENTS + GROUP_B_AGENTS:
        if not os.path.isfile(spec):
            raise FileNotFoundError(f"Agent file not found: {spec}")
    wandb.init(
        entity="marco-wojtek-tu-dortmund",
        project="Evaluation",
    )
    try:
        print("Cross-group evaluation: Belief-state agents vs Non-belief-state agents")
        run_cross_group_evaluation()
    except Exception as e:
        print(f"Error during cross-group evaluation: {e}")
