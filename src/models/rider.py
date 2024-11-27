from typing import Tuple

class Rider:
    def __init__(self, pickup_location: Tuple[float, float], dropoff_location: Tuple[float, float]):
        self.pickup_location = pickup_location
        self.dropoff_location = dropoff_location
        self.assigned_driver = None
        self.effective_supply = 0.0
