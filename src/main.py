from models.road_network import create_sample_road_network
from models.cluster import ClusterPoint, OffDutyDistribution
from simulation.simulation import RideHailSimulation
from visualization.visualizer import RideHailVisualization
from simulation.config import TOTAL_DRIVERS
import numpy as np

def main():
    # Create road network with larger size
    MAP_SIZE = 50  # Increased from 20
    road_network = create_sample_road_network(size=MAP_SIZE)
    
    # Define off-duty clusters near map edges
    clusters = [
        # Corner clusters
        ClusterPoint(
            center=(5, 5),  # Adjusted for larger map
            covariance=np.array([[2.0, 0], [0, 2.0]]),  # Increased spread
            weight=1.0
        ),
        # ... [rest of clusters] ...
    ]

    off_duty_dist = OffDutyDistribution(clusters)
    
    # Initialize simulation
    sim = RideHailSimulation(road_network, off_duty_dist, n_drivers=TOTAL_DRIVERS)
    
    # Create and show visualization
    vis = RideHailVisualization(sim)
    vis.show()

if __name__ == "__main__":
    main()
