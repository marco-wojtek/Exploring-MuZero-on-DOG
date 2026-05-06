import sys, os

import wandb
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'  # Verhindert, dass JAX den gesamten GPU-Speicher belegt, damit mehrere eval Prozesse laufen können
import chex
import jax
import jax.numpy as jnp
from time import time
from functools import partial
import pickle
import numpy as np
import math
import itertools
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
from DOG.dog import *
from MuZero_DOG.muzero_dog import *
from MuZero_DOG.eval_config.eval_with_belief import evaluate_agent_parallel, fairness_check
# from MuZero_DOG.eval_config.eval_without_belief import evaluate_agent_parallel, fairness_check
# from MuZero_DOG.evaluate_agent import evaluate_agent_parallel, fairness_check

################################################################################
# Big Evaluation: systematically evaluates all agents against each other and
# against the baseline agents (random, rule_based, untrained MuZero).
#
# AGENT ROSTER
# ------------
# Entries are either:
#   - a path string to a .pkl param file  (relative to project root)
#   - 'random_agent'
#   - 'rule_based_agent'
#   - None  (untrained / freshly-initialised MuZero)
#
# Since MADN uses teams (players 0&2 vs 1&3), each matchup is:
#   Team A = (agent_i as player 0, agent_i as player 2)
#   Team B = (agent_j as player 1, agent_j as player 3)
################################################################################

FOLDER = "MuZero_DOG/models/params/"
BATCH_SIZE = 250   # parallel games per evaluate_agent_parallel call

# ---------------------------------------------------------------------------
# Define which agents to include in the big evaluation.
# Add or remove entries here.
# ---------------------------------------------------------------------------
TRAINED_AGENTS = [
    # f"{FOLDER}muzero_dog_params_lr0.001_g500_it100_seed25.pkl", # session 
    # f"{FOLDER}muzero_dog_params_lr0.001_g500_it100_seed27.pkl", # session
    # f"{FOLDER}muzero_dog_params_lr0.001_g500_it100_seed28.pkl", # classic
    # f"{FOLDER}muzero_dog_params_lr0.001_g500_it100_seed36.pkl", # classic
    # f"{FOLDER}muzero_dog_params_lr0.001_g500_it100_seed33.pkl", # sample
    f"{FOLDER}muzero_dog_params_lr0.001_g500_it50_seed37.pkl", # sample
]

# TRAINED_AGENTS = [
#     # f"{FOLDER}muzero_dog_params_lr0.001_g500_it100_seed23.pkl", # session 
#     # f"{FOLDER}muzero_dog_params_lr0.001_g500_it100_seed24.pkl", # session
#     # f"{FOLDER}muzero_dog_params_lr0.001_g500_it100_seed40.pkl", # classic
#     f"{FOLDER}muzero_dog_params_lr0.001_g500_it100_seed41.pkl", # classic
#     f"{FOLDER}muzero_dog_params_lr0.001_g500_it100_seed22.pkl", # sample
#     f"{FOLDER}muzero_dog_params_lr0.001_g500_it100_seed39.pkl", # sample
# ]

# 0: random, 1: rule-based, 2: classic, 3:sample-based, 4: session-based
TRAINED_TYPES = [
    # 4,  
    # 4,
    # 2,
    2,
    # 3,
    # 3,
]

BASELINE_AGENTS = [
    'random_agent',
    'rule_based_agent',
    None,   # untrained MuZero
]
BASELINE_TYPES = [
    None,   # random_agent doesn't need a type
    None,   # rule_based_agent doesn't need a type
    None,   # untrained MuZero doesn't need a type
]

def short_name(agent_spec) -> str:
    """Return a short human-readable label for an agent spec."""
    if agent_spec is None:
        return "untrained"
    if isinstance(agent_spec, str) and agent_spec in ('random_agent', 'rule_based_agent'):
        return agent_spec
    # file path → extract filename without extension
    return os.path.splitext(os.path.basename(agent_spec))[0]


def load_agent(agent_spec):
    """Load params from file or return a sentinel string / None for special agents."""
    if agent_spec is None or agent_spec in ('random_agent', 'rule_based_agent'):
        return agent_spec
    return load_params_from_file(agent_spec)


def run_matchup(spec_a, spec_b, type_a=None, type_b=None, batch_size=BATCH_SIZE):
    """
    Run one team matchup: Team A (spec_a) vs Team B (spec_b).
    Returns the raw outputs of evaluate_agent_parallel.
    Players 0 & 2 use spec_a; players 1 & 3 use spec_b.
    """
    params_a = load_agent(spec_a)
    params_b = load_agent(spec_b)
    return evaluate_agent_parallel(params_a, params_b, params_a, params_b,
                                   type1=type_a, type2=type_b, type3=type_a, type4=type_b,
                                   batch_size=batch_size)


def run_big_evaluation(
    trained_agents=None,
    baseline_agents=None,
    batch_size=BATCH_SIZE,
    seed=None,
    skip_symmetric=True,
):
    """
    Systematically evaluate every agent against every other agent.

    Phase 1 – Each trained agent vs every baseline (random, rule_based, untrained).
    Phase 2 – Every pair of trained agents faces off against each other.
              If skip_symmetric=True, (A vs B) is played once; (B vs A) is
              skipped since roles are symmetric in team play.

    Args:
        trained_agents : list of agent specs (file paths).  Defaults to TRAINED_AGENTS.
        baseline_agents: list of baseline specs.             Defaults to BASELINE_AGENTS.
        batch_size      : number of parallel games per matchup.
        seed            : optional fixed seed for reproducibility.
        skip_symmetric  : if True, only play each unordered pair once
                          (team A vs B, not also B vs A separately).
    """
    if trained_agents is None:
        trained_agents = TRAINED_AGENTS
    if baseline_agents is None:
        baseline_agents = BASELINE_AGENTS

    results = {}   # key: (name_a, name_b) -> raw evaluate_agent_parallel output
    total_start = time()

    # ------------------------------------------------------------------
    # Phase 1: trained vs baseline
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PHASE 1 – Trained agents vs baselines")
    print("=" * 70)

    for agent_spec, agent_type in zip(trained_agents, TRAINED_TYPES):
        for baseline_spec, baseline_type in zip(baseline_agents, BASELINE_TYPES):
            name_a = short_name(agent_spec)
            name_b = short_name(baseline_spec)

            # None (untrained) agent mirrors the trained agent's architecture
            # so the comparison is fair (same capacity, just no training).
            effective_baseline_type = agent_type if (baseline_spec is None and baseline_type is None) else baseline_type

            print(f"\n[{name_a}]  vs  [{name_b}]")
            t0 = time()
            result = run_matchup(agent_spec, baseline_spec, type_a=agent_type, type_b=effective_baseline_type, batch_size=batch_size)
            elapsed = time() - t0
            print(f"  → done in {elapsed:.1f}s")

            results[(name_a, name_b)] = result

    # ------------------------------------------------------------------
    # Phase 2: trained vs trained (all combinations / pairs)
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PHASE 2 – Trained agents vs trained agents")
    print("=" * 70)

    if skip_symmetric:
        pairs = list(itertools.combinations(range(len(trained_agents)), 2))
    else:
        pairs = list(itertools.permutations(range(len(trained_agents)), 2))

    for i, j in pairs:
        spec_a = trained_agents[i]
        spec_b = trained_agents[j]
        name_a = short_name(spec_a)
        name_b = short_name(spec_b)

        print(f"\n[{name_a}]  vs  [{name_b}]")
        t0 = time()
        result = run_matchup(spec_a, spec_b, type_a=TRAINED_TYPES[i], type_b=TRAINED_TYPES[j], batch_size=batch_size)
        elapsed = time() - t0
        print(f"  → done in {elapsed:.1f}s")

        results[(name_a, name_b)] = result

    total_elapsed = time() - total_start
    print("\n" + "=" * 70)
    print(f"Big evaluation complete in {total_elapsed:.1f}s  "
          f"({len(results)} matchups)")
    print("=" * 70)

    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    
    wandb.init(
        entity="marco-wojtek-tu-dortmund",
        project="Evaluation",
    )
    try:
        # check existence of all trained agent files before starting the big evaluation
        for agent_spec in TRAINED_AGENTS:
            if agent_spec is not None and not os.path.isfile(agent_spec):
                raise FileNotFoundError(f"Trained agent file not found: {agent_spec}")
        # print("FIRST RANDOM AGENT GAME")
        # fairness_check(100_000)
        # print("Evaluating agents for DOG with belief states enabled...")
        print(("Evaluate only Classic which trained with 100 simulations"))
        run_big_evaluation()
    except Exception as e:
        print(f"Error during big evaluation: {e}")

