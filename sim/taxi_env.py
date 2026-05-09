import random
from features.demand_hotspots import generate_hotspot_passenger


class TaxiDispatchEnv:
    """
    Smart Taxi Dispatch Environment with:
    - Demand hotspots
    - Taxi repositioning
    - Q-learning support
    """

    def __init__(self, grid_size=5, num_taxis=3, max_steps=50, seed=42):

        self.grid_size = grid_size
        self.num_taxis = num_taxis

        # Extra action for repositioning
        self.num_actions = num_taxis + 1

        self.max_steps = max_steps
        self.seed = seed

        random.seed(seed)

        self.taxis = []
        self.pickup = None
        self.drop = None
        self.current_step = 0

    def reset(self):

        self.current_step = 0

        self.taxis = [
            self._random_location()
            for _ in range(self.num_taxis)
        ]

        self.pickup, self.drop = self._generate_passenger()

        return self._get_state()

    def step(self, action):

        if action < 0 or action >= self.num_actions:
            raise ValueError("Invalid taxi action selected")

        # =========================================
        # REPOSITION ACTION
        # =========================================
        if action == self.num_taxis:

            hotspot = (2, 2)

            updated_positions = []

            for taxi in self.taxis:

                x, y = taxi
                hx, hy = hotspot

                # Move taxi 1 step toward hotspot
                if x < hx:
                    x += 1
                elif x > hx:
                    x -= 1

                if y < hy:
                    y += 1
                elif y > hy:
                    y -= 1

                updated_positions.append((x, y))

            self.taxis = updated_positions

            waiting_time = 1
            reward = -1

            selected_taxi = "reposition"

        # =========================================
        # NORMAL DISPATCH ACTION
        # =========================================
        else:

            selected_taxi_location = self.taxis[action]

            empty_distance = self._manhattan_distance(
                selected_taxi_location,
                self.pickup
            )

            waiting_time = empty_distance

            reward = -waiting_time

            # Taxi completes ride
            self.taxis[action] = self.drop

            selected_taxi = action

        # Generate next passenger
        self.pickup, self.drop = self._generate_passenger()

        self.current_step += 1

        done = self.current_step >= self.max_steps

        next_state = self._get_state()

        info = {
            "selected_taxi": selected_taxi,
            "waiting_time": waiting_time,
            "taxi_positions": self.taxis,
            "pickup": self.pickup,
            "drop": self.drop
        }

        return next_state, reward, done, info

    def _get_state(self):

        distances = [
            self._manhattan_distance(taxi, self.pickup)
            for taxi in self.taxis
        ]

        state = tuple(
            distances + [self.pickup[0], self.pickup[1]]
        )

        return state

    def _generate_passenger(self):

        return generate_hotspot_passenger(self.grid_size)

    def _random_location(self):

        return (
            random.randint(0, self.grid_size - 1),
            random.randint(0, self.grid_size - 1)
        )

    def _manhattan_distance(self, loc1, loc2):

        return (
            abs(loc1[0] - loc2[0]) +
            abs(loc1[1] - loc2[1])
        )


# =========================================
# TEST ENVIRONMENT
# =========================================

if __name__ == "__main__":

    env = TaxiDispatchEnv(
        grid_size=5,
        num_taxis=3,
        max_steps=5
    )

    state = env.reset()

    print("Initial State:", state)
    print("Taxi Positions:", env.taxis)

    done = False

    while not done:

        action = random.randint(0, env.num_actions - 1)

        next_state, reward, done, info = env.step(action)

        print("\nAction:", action)
        print("Next State:", next_state)
        print("Reward:", reward)
        print("Info:", info)