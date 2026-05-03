# 🚕 Smart Taxi Dispatch Optimization using Reinforcement Learning and MLOps

## 📌 Problem Statement
The objective of this project is to optimize taxi dispatch decisions in a city grid to reduce passenger waiting time and minimize empty taxi travel distance.

---

## 🌍 SDG Link
This project supports **SDG 11 – Sustainable Cities and Communities** by:
- Reducing passenger waiting time  
- Improving urban transportation efficiency  
- Minimizing congestion and fuel wastage  

---

## 🧪 Simulator
The simulator models a city as a **5×5 grid**:
- Multiple taxis are randomly placed on the grid  
- A passenger request (pickup + drop) is generated dynamically  
- The RL agent decides which taxi to dispatch  

---

## 🧠 RL Algorithm
We use **Q-learning** because:
- The state space is discrete  
- The action space is small and finite  
- It is simple and efficient for this environment  

---

## 📊 State Representation

```text
(distance_taxi_0, distance_taxi_1, distance_taxi_2, pickup_x, pickup_y)
