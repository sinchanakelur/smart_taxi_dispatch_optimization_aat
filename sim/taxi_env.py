import random


class TaxiDispatchEnv:
    """
    Working simulator for Smart Taxi Dispatch Optimization.

    Environment:
    - City is represented as a grid.
    - Multiple taxis are placed randomly.
    - One passenger request is generated at a time.
    - RL agent selects which taxi to dispatch.
    """

    def __init__(self, grid_size=5, num_taxis=3, max_steps=50, seed=42):
        self.grid_size = grid_size
        self.num_taxis = num_taxis
        self.max_steps = max_steps
        self.seed = seed

        random.seed(seed)

        self.taxis = []
        self.pickup = None
        self.drop = None
        self.current_step = 0

    def reset(self):
        """
        Resets simulator for a new episode.
        """
        self.current_step = 0

        self.taxis = [
            self._random_location()
            for _ in range(self.num_taxis)
        ]

        self.pickup, self.drop = self._generate_passenger()

        return self._get_state()

    def step(self, action):
        """
        Performs one simulator step.

        action:
        - 0 means dispatch taxi 0
        - 1 means dispatch taxi 1
        - 2 means dispatch taxi 2
        """

        if action < 0 or action >= self.num_taxis:
            raise ValueError("Invalid taxi action selected")

        selected_taxi_location = self.taxis[action]

        # Passenger waiting time is approximated as distance
        # from selected taxi to passenger pickup location.
        empty_distance = self._manhattan_distance(
            selected_taxi_location,
            self.pickup
        )

        waiting_time = empty_distance

        # Reward is negative because RL tries to maximize reward.
        reward = -waiting_time

        # Taxi moves to passenger pickup first,
        # then completes trip and ends at drop location.
        self.taxis[action] = self.drop

        # Generate next passenger request
        self.pickup, self.drop = self._generate_passenger()

        self.current_step += 1
        done = self.current_step >= self.max_steps

        next_state = self._get_state()

        info = {
            "selected_taxi": action,
            "empty_distance": empty_distance,
            "waiting_time": waiting_time,
            "taxi_positions": self.taxis,
            "pickup": self.pickup,
            "drop": self.drop
        }

        return next_state, reward, done, info

    def _get_state(self):
        """
        Improved state:
        - distance of taxi 0 to pickup
        - distance of taxi 1 to pickup
        - distance of taxi 2 to pickup
        - pickup x
        - pickup y
        """

        distances = [
            self._manhattan_distance(taxi, self.pickup)
            for taxi in self.taxis
        ]

        state = tuple(distances + [self.pickup[0], self.pickup[1]])

        return state

    def _generate_passenger(self):
        """
        Generates random pickup and drop locations.
        Pickup and drop should not be the same.
        """

        pickup = self._random_location()
        drop = self._random_location()

        while drop == pickup:
            drop = self._random_location()

        return pickup, drop

    def _random_location(self):
        """
        Generates a random grid cell.
        """

        return (
            random.randint(0, self.grid_size - 1),
            random.randint(0, self.grid_size - 1)
        )

    def _manhattan_distance(self, loc1, loc2):
        """
        Manhattan distance for grid movement.
        """

        return abs(loc1[0] - loc2[0]) + abs(loc1[1] - loc2[1])


# Test simulator directly
if __name__ == "__main__":
    env = TaxiDispatchEnv(grid_size=5, num_taxis=3, max_steps=5)

    state = env.reset()
    print("Initial State:", state)
    print("Initial Taxi Positions:", env.taxis)
    print("Initial Pickup:", env.pickup)
    print("Initial Drop:", env.drop)

    done = False

    while not done:
        action = random.randint(0, env.num_taxis - 1)
        next_state, reward, done, info = env.step(action)

        print("\nAction:", action)
        print("Next State:", next_state)
        print("Reward:", reward)
        print("Done:", done)
        print("Info:", info)