# Lab 1: The Multi-Armed Bandit Problem and MDP Foundations

MSDS 684 — Reinforcement Learning | Morgan Cooper

## Overview

This lab explores the multi-armed bandit problem and Markov decision process (MDP) foundations using PyTorch and Gymnasium.

## Contents

- `part1_multi_armed_bandits.ipynb` — 10-armed bandit with epsilon-greedy and UCB agents
- `part2_frozen_lake.ipynb` — FrozenLake-v1 environment exploration with random agent baseline
- `part2_taxi.ipynb` — Taxi-v3 environment exploration with random agent baseline
- `Cooper_Morgan_Lab1.pdf` — Lab report

## Setup

```
conda create -n rl-bandits python=3.11 -y
conda activate rl-bandits
pip install torch numpy matplotlib gymnasium ipykernel
```

## Key Results

| Strategy | Reward (step 2000) | % Optimal (step 2000) |
|---|---|---|
| ε=0.01 | 1.3734 | 69.10% |
| ε=0.1 | 1.3581 | 82.30% |
| ε=0.2 | 1.1829 | 74.60% |
| UCB c=1 | 1.5000 | 94.70% |
| UCB c=2 | 1.4736 | 89.80% |
