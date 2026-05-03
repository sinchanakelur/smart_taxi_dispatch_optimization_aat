# RL Methodology

## Algorithm Choice
This project uses Q-learning because the taxi dispatch environment has discrete states and a small discrete action space.

## State
The state represents the distance of each taxi from the passenger pickup point and the pickup location.

State format:

```text
(distance_taxi_0, distance_taxi_1, distance_taxi_2, pickup_x, pickup_y)