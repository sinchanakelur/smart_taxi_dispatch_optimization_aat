from fastapi import FastAPI
import joblib
import numpy as np
import mlflow

app = FastAPI(
    title="Smart Taxi Dispatch API"
)

q_table = joblib.load("policies/policy_v2_explored.pkl")


@app.get("/")
def home():
    return {
        "message": "Smart Taxi Dispatch RL API Running"
    }


@app.post("/predict_taxi")
def predict_taxi(
    taxi0_distance: int,
    taxi1_distance: int,
    taxi2_distance: int,
    pickup_x: int,
    pickup_y: int
):

    state = (
        taxi0_distance,
        taxi1_distance,
        taxi2_distance,
        pickup_x,
        pickup_y
    )

    if state not in q_table:

        distances = [
            taxi0_distance,
            taxi1_distance,
            taxi2_distance
        ]

        action = distances.index(min(distances))

        mlflow.log_param("pickup_x", pickup_x)
        mlflow.log_param("pickup_y", pickup_y)
        mlflow.log_metric("selected_taxi", action)

        return {
            "selected_taxi": action,
            "note": "Nearest taxi selected using fallback logic"
        }

    action = int(np.argmax(q_table[state]))

    mlflow.log_param("pickup_x", pickup_x)
    mlflow.log_param("pickup_y", pickup_y)
    mlflow.log_metric("selected_taxi", action)

    return {
        "selected_taxi": action
    }