import numpy as np
import networkx as nx
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as patches
from typing import Tuple, List, Optional
from enum import Enum
import time
import random
from sklearn.cluster import KMeans
from scipy.stats import poisson

# Simulation parameters
TOTAL_DRIVERS = 20  # Total number of drivers in simulation
RIDER_GENERATION_RATE = 1.0  # Average number of new riders per time unit
N_RIDER_CLUSTERS = 5  # Number of demand hotspots
RIDER_CLUSTER_STDDEV = 2.0  # How spread out riders are around hotspots
N_DROPOFF_CLUSTERS = 5  # Number of common dropoff locations
DROPOFF_CLUSTER_STDDEV = 1.5  # How concentrated dropoffs are around centers
M_NEAREST_RIDERS = 3  # Number of nearest riders each driver considers
DRIVER_STATE_CHANGE_RATE = 0.1  # Base rate for driver state changes

class ClusterPoint:
    def __init__(self, center: Tuple[float, float], covariance: np.ndarray, weight: float):
        self.center = center
        self.covariance = covariance
        self.weight = weight
        # Create the multivariate normal distribution for this cluster
        self.distribution = multivariate_normal(mean=center, cov=covariance)
    
    def get_density(self, point: Tuple[float, float]) -> float:
        """Calculate the probability density at a given point"""
        return self.weight * self.distribution.pdf(point)
    
    def sample(self) -> Tuple[float, float]:
        """Generate a random point from this cluster's distribution"""
        return tuple(self.distribution.rvs())

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

class Rider:
    def __init__(self, pickup_location: Tuple[float, float], dropoff_location: Tuple[float, float]):
        self.pickup_location = pickup_location
        self.dropoff_location = dropoff_location
        self.assigned_driver = None
        self.effective_supply = 0.0

class OffDutyDistribution:
    def __init__(self, clusters: List[ClusterPoint]):
        self.clusters = clusters
    
    def get_density(self, point: Tuple[float, float]) -> float:
        """Calculate the total probability density at a given point"""
        return sum(cluster.get_density(point) for cluster in self.clusters)
    
    def sample_point(self) -> Tuple[float, float]:
        """Generate a random point from the mixture distribution"""
        # Choose a random cluster based on weights
        weights = [cluster.weight for cluster in self.clusters]
        total_weight = sum(weights)
        normalized_weights = [w/total_weight for w in weights]
        chosen_cluster = random.choices(self.clusters, weights=normalized_weights)[0]
        
        # Sample from the chosen cluster
        return chosen_cluster.sample()

class RideHailSimulation:
    def __init__(self, road_network, off_duty_dist, n_drivers):
        # Store basic simulation parameters
        self.road_network = road_network
        self.off_duty_distribution = off_duty_dist
        
        # Initialize collections
        self.drivers = []
        self.riders = []
        self.time = 0.0
        
        # Create initial drivers
        for _ in range(n_drivers):
            # Random starting position from road network nodes
            start_pos = random.choice(list(road_network.nodes()))
            driver = Driver(
                location=start_pos,
                state=DriverState.CRUISING
            )
            self.drivers.append(driver)
        
        # Add new attributes for rider generation
        self.rider_generation_rate = RIDER_GENERATION_RATE
        # Generate random demand centers that change periodically
        self.update_demand_centers()
        self.last_demand_update = self.time
        self.demand_update_interval = 30.0  # Time between demand center updates
        
        # Add dropoff cluster initialization
        self.update_dropoff_centers()
        self.last_dropoff_update = self.time
        self.dropoff_update_interval = 60.0  # Time between dropoff center updates
    
    def update_demand_centers(self):
        """Create new random demand centers"""
        nodes = list(self.road_network.nodes())
        self.demand_centers = random.sample(nodes, N_RIDER_CLUSTERS)
    
    def update_dropoff_centers(self):
        """Create new dropoff centers, biased towards map center"""
        max_coord = self.road_network.number_of_nodes()**0.5
        center = max_coord / 2
        
        # Generate clusters with bias towards center
        self.dropoff_centers = []
        for _ in range(N_DROPOFF_CLUSTERS):
            # Generate center point with bias towards map center
            x = np.random.normal(center, max_coord/4)
            y = np.random.normal(center, max_coord/4)
            # Ensure coordinates are within map bounds
            x = max(0, min(x, max_coord))
            y = max(0, min(y, max_coord))
            self.dropoff_centers.append((x, y))
    
    def generate_dropoff_location(self) -> Tuple[float, float]:
        """Generate a dropoff location based on clusters"""
        # Randomly select a dropoff center
        center = random.choice(self.dropoff_centers)
        
        # Add random offset using normal distribution
        x = center[0] + np.random.normal(0, DROPOFF_CLUSTER_STDDEV)
        y = center[1] + np.random.normal(0, DROPOFF_CLUSTER_STDDEV)
        
        # Ensure coordinates are within map bounds
        max_coord = self.road_network.number_of_nodes()**0.5
        x = max(0, min(x, max_coord))
        y = max(0, min(y, max_coord))
        
        return (x, y)
    
    def generate_riders(self, dt: float):
        """Probabilistically generate new riders"""
        # Update demand centers periodically
        if self.time - self.last_demand_update > self.demand_update_interval:
            self.update_demand_centers()
            self.last_demand_update = self.time
        
        # Update dropoff centers periodically
        if self.time - self.last_dropoff_update > self.dropoff_update_interval:
            self.update_dropoff_centers()
            self.last_dropoff_update = self.time
        
        # Use Poisson distribution to determine number of new riders
        n_new_riders = np.random.poisson(self.rider_generation_rate * dt)
        
        for _ in range(n_new_riders):
            # Choose a random demand center
            center = random.choice(self.demand_centers)
            
            # Generate pickup location near the chosen center
            pickup_x = center[0] + np.random.normal(0, RIDER_CLUSTER_STDDEV)
            pickup_y = center[1] + np.random.normal(0, RIDER_CLUSTER_STDDEV)
            pickup_loc = (pickup_x, pickup_y)
            
            # Generate dropoff location using new cluster-based method
            dropoff_loc = self.generate_dropoff_location()
            
            self.add_rider(pickup_loc, dropoff_loc)
    
    def calculate_supply_demand_balance(self) -> float:
        """
        Calculate supply-demand balance and distribute driver supply to nearby riders.
        Returns the supply-demand difference (positive means excess supply).
        """
        # Only count unmatched riders as demand
        total_demand = len([r for r in self.riders if not r.assigned_driver])
        
        # Only count cruising drivers as available supply
        cruising_drivers = [d for d in self.drivers if d.state == DriverState.CRUISING]
        total_supply = len(cruising_drivers)
        
        # Calculate supply distribution for each unmatched rider
        unmatched_riders = [r for r in self.riders if not r.assigned_driver]
        for rider in unmatched_riders:
            rider.effective_supply = 0.0  # Reset supply
            
            if not cruising_drivers:
                continue
            
            # Calculate distances to all cruising drivers
            driver_distances = []
            for driver in cruising_drivers:
                dist = np.linalg.norm(
                    np.array(driver.location) - np.array(rider.pickup_location)
                )
                driver_distances.append((driver, dist))
            
            # Sort by distance and take M nearest
            driver_distances.sort(key=lambda x: x[1])
            nearest_drivers = driver_distances[:M_NEAREST_RIDERS]
            
            # Calculate inverse distance weights
            if nearest_drivers:
                weights = [1.0 / (d[1] + 1e-6) for d in nearest_drivers]  # Add small epsilon to avoid division by zero
                weight_sum = sum(weights)
                if weight_sum > 0:
                    normalized_weights = [w / weight_sum for w in weights]
                    # Distribute one unit of supply among M nearest riders
                    for (_, _), weight in zip(nearest_drivers, normalized_weights):
                        rider.effective_supply += weight
        
        return total_supply - total_demand
    
    def update_driver_states(self, dt: float, supply_demand_diff: float):
        """Update driver states based on supply-demand balance"""
        # Adjust state change rate based on supply-demand difference
        rate = abs(supply_demand_diff) * DRIVER_STATE_CHANGE_RATE * dt
        
        # Generate number of state changes using Poisson distribution
        n_changes = np.random.poisson(rate)
        
        if n_changes == 0:
            return
            
        if supply_demand_diff > 0:  # Excess supply: some drivers go off duty
            # Get cruising drivers
            available_drivers = [d for d in self.drivers if d.state == DriverState.CRUISING]
            if available_drivers:
                # Randomly select drivers to go off duty
                for _ in range(min(n_changes, len(available_drivers))):
                    driver = random.choice(available_drivers)
                    driver.state = DriverState.OFF_DUTY
                    available_drivers.remove(driver)
        
        else:  # Supply shortage: some drivers return to duty
            # Get off-duty drivers
            off_duty_drivers = [d for d in self.drivers if d.state == DriverState.OFF_DUTY]
            if off_duty_drivers:
                # Randomly select drivers to return to duty
                for _ in range(min(n_changes, len(off_duty_drivers))):
                    driver = random.choice(off_duty_drivers)
                    driver.state = DriverState.CRUISING
                    off_duty_drivers.remove(driver)
    
    def step(self, dt: float):
        """Advance simulation by dt time steps"""
        self.time += dt
        
        # Generate new riders first
        self.generate_riders(dt)
        
        # Match available drivers with waiting riders BEFORE movement updates
        self._match_drivers_and_riders()
        
        # Calculate supply-demand balance and update driver states
        supply_demand_diff = self.calculate_supply_demand_balance()
        self.update_driver_states(dt, supply_demand_diff)
        
        # Update each driver's state and position
        for driver in self.drivers:
            if driver.state == DriverState.CRUISING:
                # Find all unmatched rider locations
                rider_locations = [
                    r.pickup_location 
                    for r in self.riders 
                    if not r.assigned_driver
                ]
                
                if rider_locations:
                    # Convert to numpy array for calculations
                    rider_positions = np.array(rider_locations)
                    current_pos = np.array(driver.location)
                    
                    if len(rider_positions) >= 5:  # Only use k-means for larger groups
                        # Use k-means to find clusters of riders
                        n_clusters = min(3, len(rider_positions))
                        kmeans = KMeans(n_clusters=n_clusters, n_init=10)
                        kmeans.fit(rider_positions)
                        
                        # Find the nearest cluster center to the driver
                        cluster_centers = kmeans.cluster_centers_
                        distances_to_clusters = np.linalg.norm(
                            cluster_centers - current_pos, 
                            axis=1
                        )
                        nearest_cluster = cluster_centers[np.argmin(distances_to_clusters)]
                        target_pos = tuple(nearest_cluster)
                    else:
                        # For small numbers of riders, just head to the nearest one
                        distances = np.linalg.norm(rider_positions - current_pos, axis=1)
                        nearest_rider_pos = rider_positions[np.argmin(distances)]
                        target_pos = tuple(nearest_rider_pos)
                else:
                    # Random exploration when no riders
                    if not hasattr(driver, 'exploration_target') or \
                       np.linalg.norm(np.array(driver.location) - np.array(driver.exploration_target)) < 0.1:
                        driver.exploration_target = random.choice(list(self.road_network.nodes()))
                    
                    target_pos = driver.exploration_target
                
                # Move towards the target
                next_pos = self._get_next_position(driver.location, target_pos)
                driver.location = next_pos
            
            elif driver.state == DriverState.ASSIGNED:
                if hasattr(driver, 'assigned_rider'):
                    pickup_location = driver.assigned_rider.pickup_location
                    next_pos = self._get_next_position(driver.location, pickup_location)
                    driver.location = next_pos
                    
                    # Use a slightly larger threshold and ensure exact arrival
                    if np.linalg.norm(np.array(driver.location) - np.array(pickup_location)) < 0.2:
                        driver.location = pickup_location  # Snap to exact location
                        driver.state = DriverState.WITH_PASSENGER
                else:
                    driver.state = DriverState.CRUISING
            
            elif driver.state == DriverState.WITH_PASSENGER:
                if hasattr(driver, 'assigned_rider'):
                    dropoff_location = driver.assigned_rider.dropoff_location
                    next_pos = self._get_next_position(driver.location, dropoff_location)
                    driver.location = next_pos
                    
                    # Check if we've reached the destination
                    if np.linalg.norm(np.array(driver.location) - np.array(dropoff_location)) < 0.2:
                        # Clean up the ride
                        rider = driver.assigned_rider
                        self.riders.remove(rider)
                        driver.assigned_rider = None
                        # Return to cruising state
                        driver.state = DriverState.CRUISING
            
            elif driver.state == DriverState.OFF_DUTY:
                # If driver doesn't have a rest destination, assign one
                if not hasattr(driver, 'rest_destination'):
                    driver.rest_destination = self.off_duty_distribution.sample_point()
                
                # Move towards rest destination
                next_pos = self._get_next_position(driver.location, driver.rest_destination)
                driver.location = next_pos
                
                # If reached destination, start rest period
                if np.linalg.norm(np.array(driver.location) - np.array(driver.rest_destination)) < 0.1:
                    if not hasattr(driver, 'rest_start_time'):
                        driver.rest_start_time = self.time
                    elif self.time - driver.rest_start_time > 30:  # 30 time units rest
                        # Return to service
                        driver.state = DriverState.CRUISING
                        delattr(driver, 'rest_destination')
                        delattr(driver, 'rest_start_time')
    
    def add_rider(self, pickup_loc, dropoff_loc):
        """Add a new rider to the simulation"""
        rider = Rider(
            pickup_location=pickup_loc,
            dropoff_location=dropoff_loc
        )
        self.riders.append(rider)
        return rider
    
    def _match_drivers_and_riders(self):
        """Match available drivers with waiting riders based on shortest travel time"""
        # Find cruising drivers and unmatched riders
        available_drivers = [d for d in self.drivers if d.state == DriverState.CRUISING]
        unmatched_riders = [r for r in self.riders if not r.assigned_driver]
        
        if not available_drivers or not unmatched_riders:
            return
        
        # Calculate travel times between all drivers and riders
        travel_times = []
        for driver in available_drivers:
            for rider in unmatched_riders:
                # Calculate estimated travel time (using Manhattan distance for now)
                # Could be replaced with actual path finding later
                distance = np.linalg.norm(
                    np.array(driver.location) - np.array(rider.pickup_location)
                )
                travel_time = distance  # Assuming unit speed for now
                travel_times.append((driver, rider, travel_time))
        
        # Sort by travel time
        travel_times.sort(key=lambda x: x[2])
        
        # Match drivers to riders, taking the shortest travel times first
        matched_drivers = set()
        matched_riders = set()
        
        for driver, rider, _ in travel_times:
            if (driver not in matched_drivers and 
                rider not in matched_riders):
                # Make the match
                driver.state = DriverState.ASSIGNED
                driver.assigned_rider = rider
                rider.assigned_driver = driver
                
                # Add to matched sets
                matched_drivers.add(driver)
                matched_riders.add(rider)
    
    def _get_nearest_node(self, position):
        """Find the nearest node in the road network to a given position"""
        nodes = np.array(list(self.road_network.nodes()))
        distances = np.linalg.norm(nodes - np.array(position), axis=1)
        return tuple(nodes[np.argmin(distances)])  # Convert back to tuple for NetworkX
    
    def _get_next_position(self, current_pos, target_pos):
        """Calculate next position along path to target, restricted to grid movements"""
        # If very close to target, just arrive there
        if np.linalg.norm(np.array(target_pos) - np.array(current_pos)) < 0.2:
            return target_pos
        
        # Convert positions to numpy arrays for easier calculation
        current = np.array(current_pos)
        target = np.array(target_pos)
        direction = target - current
        
        # Determine which direction (horizontal or vertical) has the larger difference
        abs_diff = np.abs(direction)
        step_size = 0.1
        
        if abs_diff[0] > abs_diff[1]:
            # Move horizontally
            if direction[0] > 0:
                return (current_pos[0] + step_size, current_pos[1])  # Right
            else:
                return (current_pos[0] - step_size, current_pos[1])  # Left
        else:
            # Move vertically
            if direction[1] > 0:
                return (current_pos[0], current_pos[1] + step_size)  # Up
            else:
                return (current_pos[0], current_pos[1] - step_size)  # Down

class RideHailVisualization:
    def __init__(self, simulation: RideHailSimulation, update_interval: int = 50):
        self.sim = simulation
        self.update_interval = update_interval
        # Add window size for moving average at the start
        self.window_size = 10  # Adjust this value to control smoothing amount
        
        # Setup the figure with two subplots side by side
        self.fig = plt.figure(figsize=(20, 8))
        self.map_ax = self.fig.add_subplot(121)  # Left subplot for map
        self.stats_ax = self.fig.add_subplot(122)  # Right subplot for statistics
        self.fig.set_facecolor('white')
        
        # Initialize statistics tracking
        self.times = []
        self.with_passenger_counts = []
        self.rider_counts = []
        self.cruising_counts = []
        self.off_duty_counts = []
        self.start_time = time.time()
        
        # Create color map for density
        colors = [(1, 1, 1), (0.9, 0.9, 1), (0.7, 0.7, 1), (0.5, 0.5, 1)]
        self.density_cmap = LinearSegmentedColormap.from_list('density', colors)
        
        # Setup the density plot on map subplot
        self.setup_density_plot()
        
        # Plot road network
        self.plot_road_network()
        
        # Initialize driver plots
        self.driver_plots = {}
        self.setup_driver_plots()
        
        # Setup legend
        self.setup_legend()
        
        # Add rider scatter plot
        self.rider_plot = self.map_ax.scatter(
            [], [], marker='x', c='black', s=100, 
            label='Waiting Riders', zorder=4
        )
        
        # Setup statistics plot
        self.setup_stats_plot()
        
        # Setup animation
        self.anim = FuncAnimation(
            self.fig, self.update, interval=update_interval, 
            frames=None, blit=True
        )
    
    def setup_density_plot(self):
        # Create grid for density visualization
        x = np.linspace(0, self.sim.road_network.number_of_nodes()**0.5, 100)
        y = np.linspace(0, self.sim.road_network.number_of_nodes()**0.5, 100)
        X, Y = np.meshgrid(x, y)
        positions = np.vstack([X.ravel(), Y.ravel()]).T
        
        # Calculate density
        Z = np.zeros(len(positions))
        for i, pos in enumerate(positions):
            Z[i] = self.sim.off_duty_distribution.get_density(pos)
        
        # Normalize and reshape
        Z = Z.reshape(X.shape)
        Z = (Z - Z.min()) / (Z.max() - Z.min())
        
        # Plot density
        self.density_plot = self.map_ax.imshow(
            Z, extent=[x.min(), x.max(), y.min(), y.max()],
            cmap=self.density_cmap, alpha=0.3, zorder=1
        )
    
    def plot_road_network(self):
        # Plot edges
        for edge in self.sim.road_network.edges():
            x = [edge[0][0], edge[1][0]]
            y = [edge[0][1], edge[1][1]]
            self.map_ax.plot(x, y, 'gray', linewidth=0.5, alpha=0.5, zorder=2)
    
    def setup_driver_plots(self):
        # Create scatter plots for each driver state
        state_colors = {
            DriverState.CRUISING: 'blue',
            DriverState.ASSIGNED: 'yellow',
            DriverState.WITH_PASSENGER: 'green',
            DriverState.OFF_DUTY: 'red'
        }
        
        for state in DriverState:
            self.driver_plots[state] = self.map_ax.scatter(
                [], [], c=state_colors[state], label=state.value,
                s=100, zorder=3
            )
    
    def setup_legend(self):
        self.map_ax.legend(loc='upper right')
        self.map_ax.set_title('Ride-hailing Simulation')
        self.map_ax.set_xlabel('X coordinate')
        self.map_ax.set_ylabel('Y coordinate')
    
    def setup_stats_plot(self):
        """Initialize the statistics subplot"""
        self.stats_ax.set_title('System Statistics')
        self.stats_ax.set_xlabel('Time (s)')
        self.stats_ax.set_ylabel('Count')
        
        # Initialize empty lines for each metric
        self.with_passenger_line, = self.stats_ax.plot(
            [], [], 'g-', label='Drivers with Passengers'
        )
        self.rider_line, = self.stats_ax.plot(
            [], [], 'k-', label='Waiting Riders'
        )
        self.cruising_line, = self.stats_ax.plot(
            [], [], 'b-', label='Cruising Drivers'
        )
        self.off_duty_line, = self.stats_ax.plot(
            [], [], 'r-', label='Off-duty Drivers'
        )
        
        self.stats_ax.legend()
        self.stats_ax.grid(True)
        
        # Set initial axis limits
        self.stats_ax.set_xlim(0, 60)  # Show last 60 seconds
        self.stats_ax.set_ylim(0, 50)  # Adjust based on your expected ranges
    
    def update_stats(self):
        """Update statistics tracking"""
        current_time = time.time() - self.start_time
        self.times.append(current_time)
        
        # Count drivers in each state
        with_passenger = len([d for d in self.sim.drivers if d.state == DriverState.WITH_PASSENGER])
        cruising = len([d for d in self.sim.drivers if d.state == DriverState.CRUISING])
        off_duty = len([d for d in self.sim.drivers if d.state == DriverState.OFF_DUTY])
        waiting_riders = len([r for r in self.sim.riders if not r.assigned_driver])
        
        # Append to tracking lists
        self.with_passenger_counts.append(with_passenger)
        self.rider_counts.append(waiting_riders)
        self.cruising_counts.append(cruising)
        self.off_duty_counts.append(off_duty)
        
        # Remove old data points (keep last 60 seconds)
        while self.times and self.times[0] < current_time - 60:
            self.times.pop(0)
            self.with_passenger_counts.pop(0)
            self.rider_counts.pop(0)
            self.cruising_counts.pop(0)
            self.off_duty_counts.pop(0)
        
        # Apply moving average smoothing
        def moving_average(data):
            if len(data) < self.window_size:
                return data
            return [sum(data[max(0, i-self.window_size):i])/min(i, self.window_size) 
                   for i in range(1, len(data)+1)]
        
        smoothed_passenger = moving_average(self.with_passenger_counts)
        smoothed_riders = moving_average(self.rider_counts)
        smoothed_cruising = moving_average(self.cruising_counts)
        smoothed_off_duty = moving_average(self.off_duty_counts)
        
        # Update the lines with smoothed data
        self.with_passenger_line.set_data(self.times, smoothed_passenger)
        self.rider_line.set_data(self.times, smoothed_riders)
        self.cruising_line.set_data(self.times, smoothed_cruising)
        self.off_duty_line.set_data(self.times, smoothed_off_duty)
        
        # Adjust x-axis limits to show last 60 seconds
        if self.times:
            self.stats_ax.set_xlim(max(0, current_time - 60), current_time)
            
            # Adjust y-axis to fit all data with some padding
            max_count = max(
                max(self.with_passenger_counts + [0]),
                max(self.rider_counts + [0]),
                max(self.cruising_counts + [0]),
                max(self.off_duty_counts + [0])
            )
            self.stats_ax.set_ylim(0, max_count * 1.1)
    
    def update(self, frame):
        # Update simulation and map visualization
        self.sim.step(0.1)
        
        # Update driver positions on map
        for state in DriverState:
            drivers = [d for d in self.sim.drivers if d.state == state]
            if drivers:
                x = [d.location[0] for d in drivers]
                y = [d.location[1] for d in drivers]
                self.driver_plots[state].set_offsets(np.c_[x, y])
            else:
                self.driver_plots[state].set_offsets(np.c_[[], []])
        
        # Update rider positions
        unmatched_riders = [r for r in self.sim.riders if not r.assigned_driver]
        if unmatched_riders:
            x = [r.pickup_location[0] for r in unmatched_riders]
            y = [r.pickup_location[1] for r in unmatched_riders]
            self.rider_plot.set_offsets(np.c_[x, y])
        else:
            self.rider_plot.set_offsets(np.c_[[], []])
        
        # Update statistics
        self.update_stats()
        
        return (tuple(self.driver_plots.values()) + 
                (self.rider_plot,) +
                (self.with_passenger_line, self.rider_line, 
                 self.cruising_line, self.off_duty_line))
    
    def show(self):
        plt.show()

def create_sample_road_network(size: int = 10) -> nx.Graph:
    # Create a simple grid network
    G = nx.grid_2d_graph(size, size)
    
    # Add edge weights (distances)
    for (u, v) in G.edges():
        G[u][v]['length'] = np.sqrt(
            (u[0] - v[0])**2 + (u[1] - v[1])**2
        )
    
    return G

if __name__ == "__main__":
    # Create road network
    road_network = create_sample_road_network(size=20)
    
    # Define off-duty clusters near map edges
    map_size = 20  # Assuming 20x20 grid from create_sample_road_network
    clusters = [
        # Corner clusters
        ClusterPoint(
            center=(2, 2),
            covariance=np.array([[1.5, 0], [0, 1.5]]),
            weight=1.0
        ),
        ClusterPoint(
            center=(18, 2),
            covariance=np.array([[1.5, 0], [0, 1.5]]),
            weight=1.0
        ),
        ClusterPoint(
            center=(2, 18),
            covariance=np.array([[1.5, 0], [0, 1.5]]),
            weight=1.0
        ),
        ClusterPoint(
            center=(18, 18),
            covariance=np.array([[1.5, 0], [0, 1.5]]),
            weight=1.0
        ),
        # Edge clusters
        ClusterPoint(
            center=(10, 1),
            covariance=np.array([[2.0, 0], [0, 1.0]]),
            weight=0.7
        ),
        ClusterPoint(
            center=(10, 19),
            covariance=np.array([[2.0, 0], [0, 1.0]]),
            weight=0.7
        ),
        ClusterPoint(
            center=(1, 10),
            covariance=np.array([[1.0, 0], [0, 2.0]]),
            weight=0.7
        ),
        ClusterPoint(
            center=(19, 10),
            covariance=np.array([[1.0, 0], [0, 2.0]]),
            weight=0.7
        ),
    ]

    off_duty_dist = OffDutyDistribution(clusters)
    
    # Initialize simulation with TOTAL_DRIVERS parameter
    sim = RideHailSimulation(road_network, off_duty_dist, n_drivers=TOTAL_DRIVERS)
    
    # Create and show visualization
    vis = RideHailVisualization(sim)
    vis.show()