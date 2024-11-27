from enum import Enum
from typing import Tuple, Optional

class DriverState(Enum):
    CRUISING = "cruising"
    ASSIGNED = "assigned"
    WITH_PASSENGER = "with passenger"
    OFF_DUTY = "off duty"

class Driver:
    def __init__(self, location: Tuple[float, float], state: DriverState):
        self.location = location
        self.state = state
        self.assigned_rider = None
