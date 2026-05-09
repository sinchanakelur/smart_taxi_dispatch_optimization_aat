import random

HOTSPOTS = [
    (2, 2),
    (3, 3),
]

def generate_hotspot_passenger(grid_size):

    if random.random() < 0.7:
        pickup = random.choice(HOTSPOTS)
    else:
        pickup = (
            random.randint(0, grid_size - 1),
            random.randint(0, grid_size - 1)
        )

    drop = (
        random.randint(0, grid_size - 1),
        random.randint(0, grid_size - 1)
    )

    while drop == pickup:
        drop = (
            random.randint(0, grid_size - 1),
            random.randint(0, grid_size - 1)
        )

    return pickup, drop