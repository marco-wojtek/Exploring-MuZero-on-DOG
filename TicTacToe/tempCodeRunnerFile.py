policy_output = run_mcts(jax.random.PRNGKey(0), env, 50)
w = policy_output.action_weights
print(w)
print(w.mean(axis=0))