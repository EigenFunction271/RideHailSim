# Simulation parameters
TOTAL_DRIVERS = 50  # Total number of drivers in simulation
RIDER_GENERATION_RATE = 2.5  # Average number of new riders per time unit
N_RIDER_CLUSTERS = 5  # Number of demand hotspots
RIDER_CLUSTER_STDDEV = 2.0  # How spread out riders are around hotspots
N_DROPOFF_CLUSTERS = 5  # Number of common dropoff locations
DROPOFF_CLUSTER_STDDEV = 1.5  # How concentrated dropoffs are around centers
M_NEAREST_RIDERS = 3  # Number of nearest riders each driver considers
DRIVER_STATE_CHANGE_RATE = 0.1  # Base rate for driver state changes
