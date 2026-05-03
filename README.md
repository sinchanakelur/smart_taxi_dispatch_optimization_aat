```md
# Smart Taxi Dispatch Optimization using Reinforcement Learning and MLOps

## Problem Statement
The objective of this project is to optimize taxi dispatch decisions in a city grid to reduce passenger waiting time and empty taxi travel distance.

## SDG Link
This project supports SDG 11 – Sustainable Cities and Communities by improving urban mobility, reducing passenger delay, reducing congestion, and minimizing fuel wastage.

## Simulator
The simulator represents a city as a 5x5 grid.  
It generates random taxi positions and one passenger request at a time.  
The RL agent selects which taxi should be dispatched.

## RL Algorithm
Q-learning is used because the state and action spaces are discrete and small.

## State
```text
(distance_taxi_0, distance_taxi_1, distance_taxi_2, pickup_x, pickup_y)

pip install -r requirements.txt
python experiments/train.py --config configs/qlearning_v1.yaml
python experiments/train.py --config configs/qlearning_v2_explored.yaml
python experiments/evaluate.py
python experiments/plot_results.py