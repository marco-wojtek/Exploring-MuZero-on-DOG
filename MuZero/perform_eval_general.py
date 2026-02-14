from evaluate_agent import play_n_randomly as play_n_randomly_deterministic
from evaluate_agent import ENABLE_TEAMS as ENABLE_TEAMS_DETERMINISTIC
from evaluate_agent_stochastic import play_n_randomly as play_n_randomly_stochastic
from evaluate_agent_stochastic import ENABLE_TEAMS as ENABLE_TEAMS_STOCHASTIC
from time import time

BATCHSIZE = 100
start_time = time()
NUM_SIMULATIONS = 100
MAX_DEPTH = 50
ENABLE_TEAMS = False
ENABLE_TEAMS_DETERMINISTIC = ENABLE_TEAMS
ENABLE_TEAMS_STOCHASTIC = ENABLE_TEAMS
SEED = 75643
if __name__ == "__main__":
    print("Evaluating deterministic agent:")
    play_n_randomly_deterministic(BATCHSIZE, SEED)
    print("\nEvaluating stochastic agent:")
    play_n_randomly_stochastic(BATCHSIZE, SEED)
end_time = time()
print(f"Evaluation completed in {end_time - start_time:.2f} seconds.")