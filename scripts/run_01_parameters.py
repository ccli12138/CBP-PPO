"""Experiment 1: parameters. Default: preview only. Add --run to train into a fresh batch.
All scientific values are explicit below; config/experiments.json lists each run.
Initialization codes:1=random50%,2=upperD/lowerC,3=allD.
Routing: original -> main_PPO.py/PPO.py; UA or GF -> main_CBP_PPO.py/CBP_PPO.py.
Variants:original=PPO, UA=uniform CBP-PPO, GF=group-local financing.
"""
from experiment_runner import run_category

COMMON_SETTINGS = {'L_num': 200,
 'epochs': 1000,
 'question': 2,
 'alpha': 0.001,
 'gamma': 0.99,
 'clip_epsilon': 0.2,
 'gae_lambda': 0.95,
 'delta': 0.5,
 'rho': 0.01,
 'ppo_epochs': 1,
 'batch_size': 1,
 'device': 'cpu',
 'threads': 1}

EXPERIMENT_SETTINGS = {'r': [2.0, 5.5, 0.1],
 'eta': [0, 0.05, 0.1, 0.2, 0.3, 0.6],
 'beta': [0, 1, 2, 3, 4, 8],
 'seed': 9202,
 'question': 2,
 'display_points': 1296,
 'unique_runs': 1116,
 'zero_eta': 'Exact baseline PPO shared across beta; no duplicate training.'}

if __name__ == '__main__':
    run_category('parameters', COMMON_SETTINGS, EXPERIMENT_SETTINGS)
