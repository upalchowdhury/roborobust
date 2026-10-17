Frozen before any test evaluation (see DECISIONS.md for the full reasoning):

* reward: alpha 50 (empty-room validation), success bonus 10, failure penalty 10, step cost 0.01, **gamma 0.999** (stability sweep, rule pre-registered)
* PPO: spec defaults (MlpPolicy 64x64, lr 3e-4, gae_lambda 0.95, clip 0.2, 10 epochs, batch 256, n_steps 1024, ent_coef 0) except gamma 0.999; 4 workers
* budget: 1M requested transitions = 1,003,520 actual (245 whole rollouts), identical for both conditions
* seeds 0-4 per condition (the earlier gamma = 0.99 runs are archived and appear only in section 4 above)
* primary comparison: final checkpoints; primary endpoint: safe arrival under `combined_stress` on the 50 held-out layouts; `configs/test_suite.yaml`
* pre-declared secondary: the ~500k-transition checkpoints of the same runs, and every single condition
