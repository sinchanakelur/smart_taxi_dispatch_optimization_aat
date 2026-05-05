# RL Methodology

## Algorithm Choice
This project uses Q-learning because the taxi dispatch environment has discrete states and a small discrete action space.

## State
The state represents the distance of each taxi from the passenger pickup point and the pickup location.

## Exploration Strategy
We use ε-greedy exploration starting at ε=1.0, decaying by 0.995 per episode 
down to a minimum of 0.01. This ensures the agent explores broadly early on 
and exploits learned knowledge in later episodes.

## Convergence Discussion
Training over 800 episodes shows the average reward improving from approximately 
-240 (early episodes) to -210 (final episodes) in v2, indicating the agent 
learns to dispatch closer taxis over time. The policy stabilizes after ~500 episodes.

State format:

```text
(distance_taxi_0, distance_taxi_1, distance_taxi_2, pickup_x, pickup_y)