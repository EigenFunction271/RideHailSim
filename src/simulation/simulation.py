import numpy as np
import random
from typing import Tuple, List
from sklearn.cluster import KMeans
from scipy.stats import poisson
import networkx as nx
import logging
import time

from models.driver import Driver, DriverState
from models.rider import Rider
from models.cluster import OffDutyDistribution
from .config import *

class RideHailSimulation:
    def __init__(self, road_network, off_duty_dist, n_drivers):
        """Initialize the simulation
        Args:
            road_network: The road network
            off_duty_dist: The off-duty distribution
            n_drivers: Number of drivers to simulate
        """
        self.road_network = road_network
        self.off_duty_dist = off_duty_dist
        
        # Initialize collections
        self.drivers = []
        self.riders = []
        self.current_time = 0.0
        
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
        self.last_demand_update = self.current_time
        self.demand_update_interval = 30.0  # Time between demand center updates
        
        # Add dropoff cluster initialization
        self.update_dropoff_centers()
        self.last_dropoff_update = self.current_time
        self.dropoff_update_interval = 60.0  # Time between dropoff center updates
        
        # Add tracking for previous positions to prevent oscillations
        self.driver_previous_positions = {}  # {driver_id: [last_3_positions]}
        self.position_history_limit = 3
    
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
        
        # Ensure coordinates are within map bounds and snap to nearest grid node
        max_coord = self.road_network.number_of_nodes()**0.5
        x = max(0, min(round(x), max_coord - 1))
        y = max(0, min(round(y), max_coord - 1))
        
        return (x, y)
    
    def generate_riders(self, dt: float):
        """Probabilistically generate new riders"""
        # Define max_coord based on the map size
        max_coord = self.road_network.size()
        
        # Update demand centers periodically
        if self.current_time - self.last_demand_update > self.demand_update_interval:
            self.update_demand_centers()
            self.last_demand_update = self.current_time
        
        # Update dropoff centers periodically
        if self.current_time - self.last_dropoff_update > self.dropoff_update_interval:
            self.update_dropoff_centers()
            self.last_dropoff_update = self.current_time
        
        # Use Poisson distribution to determine number of new riders
        n_new_riders = np.random.poisson(self.rider_generation_rate * dt)
        
        for _ in range(n_new_riders):
            # Choose a random demand center
            center = random.choice(self.demand_centers)
            
            # Generate pickup location near the chosen center and snap to grid
            pickup_x = center[0] + np.random.normal(0, RIDER_CLUSTER_STDDEV)
            pickup_y = center[1] + np.random.normal(0, RIDER_CLUSTER_STDDEV)
            pickup_x = max(0, min(round(pickup_x), max_coord - 1))
            pickup_y = max(0, min(round(pickup_y), max_coord - 1))
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
    
    def step(self, dt):
        """Advance simulation by one timestep"""
        self.current_time += dt
        
        # Create a more pronounced periodic effect with multiple frequencies
        time_factor = (
            2 * np.sin(self.current_time * 0.05) +  # Slow oscillation
            np.sin(self.current_time * 0.1) +       # Medium oscillation
            0.5 * np.sin(self.current_time * 0.2)   # Fast oscillation
        ) / 2.0 + 2  # Normalize to oscillate between 0.5 and 3.5
        
        adjusted_rate = RIDER_GENERATION_RATE * time_factor
        n_new_riders = int(adjusted_rate * dt)
        
        # Generate riders with clustered pickup locations
        for _ in range(n_new_riders):
            # Choose a random cluster center for pickup
            pickup_x = random.gauss(
                random.choice([10, 20, 30, 40]), 
                RIDER_CLUSTER_STDDEV
            )
            pickup_y = random.gauss(
                random.choice([10, 20, 30, 40]), 
                RIDER_CLUSTER_STDDEV
            )
            
            # Ensure coordinates are within bounds
            pickup_x = max(0, min(self.road_network.number_of_nodes()**0.5, pickup_x))
            pickup_y = max(0, min(self.road_network.number_of_nodes()**0.5, pickup_y))
            
            # Generate dropoff location
            dropoff_x = random.uniform(0, self.road_network.number_of_nodes()**0.5)
            dropoff_y = random.uniform(0, self.road_network.number_of_nodes()**0.5)
            
            new_rider = Rider(
                pickup_location=(pickup_x, pickup_y),
                dropoff_location=(dropoff_x, dropoff_y)
            )
            self.riders.append(new_rider)
        
        # Clean up completed rides
        self.riders = [r for r in self.riders if not (r.assigned_driver and 
                      r.assigned_driver.state == DriverState.CRUISING)]
        
        # Update driver positions and states
        for driver in self.drivers:
            self._update_driver_position(driver)
        self._update_driver_states()
    
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
                # Calculate estimated travel time using Manhattan distance
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
    
    def _get_next_position(self, current_pos, target_pos=None):
        """Calculate next position along path to target using periodic boundary conditions"""
        grid_size = int(self.road_network.number_of_nodes() ** 0.5)
        
        def wrap_coordinate(x):
            return x % grid_size

        # If no target, move randomly for cruising drivers
        if target_pos is None:
            neighbors = self._get_valid_neighbors(current_pos)
            return random.choice(neighbors) if neighbors else current_pos

        # Convert positions to numpy arrays for easier calculation
        current = np.array(current_pos)
        target = np.array(target_pos)
        direction = target - current

        # Determine which direction (horizontal or vertical) has the larger difference
        abs_diff = np.abs(direction)
        step_size = 1  # Adjusted step size for more noticeable movement

        if abs_diff[0] > abs_diff[1]:
            # Move horizontally
            if direction[0] > 0:
                new_x = wrap_coordinate(current_pos[0] + step_size)
                return (new_x, current_pos[1])  # Right
            else:
                new_x = wrap_coordinate(current_pos[0] - step_size)
                return (new_x, current_pos[1])  # Left
        else:
            # Move vertically
            if direction[1] > 0:
                new_y = wrap_coordinate(current_pos[1] + step_size)
                return (current_pos[0], new_y)  # Up
            else:
                new_y = wrap_coordinate(current_pos[1] - step_size)
                return (current_pos[0], new_y)  # Down
    
    def _is_close_enough(self, pos1, pos2, threshold=1.0):
        """Check if two positions are close enough considering periodic boundaries"""
        grid_size = int(self.road_network.number_of_nodes() ** 0.5)
        
        try:
            # Calculate differences in both directions
            dx = abs(pos1[0] - pos2[0])
            dy = abs(pos1[1] - pos2[1])
            
            # Consider wrapping around in both directions
            dx = min(dx, grid_size - dx)
            dy = min(dy, grid_size - dy)
            
            # Calculate periodic distance
            return (dx*dx + dy*dy)**0.5 < threshold
        except (ValueError, TypeError):
            return False
    
    def _apply_boundary_conditions(self, pos):
        """Apply periodic boundary conditions to keep drivers within the map"""
        x, y = pos
        size = self.road_network.size()
        
        # Wrap around the map edges
        x = x % size
        y = y % size
        
        return (x, y)
    
    def _update_driver_position(self, driver):
        """Update driver location based on their state and targets"""
        if driver.state == DriverState.CRUISING:
            # Random walk for cruising drivers
            dx = random.choice([-1, 1])
            dy = random.choice([-1, 1])
            new_x = max(0, min(self.road_network.number_of_nodes()**0.5, driver.location[0] + dx))
            new_y = max(0, min(self.road_network.number_of_nodes()**0.5, driver.location[1] + dy))
            driver.location = (new_x, new_y)
            
        elif driver.state == DriverState.ASSIGNED:
            # Move toward pickup location
            if driver.assigned_rider:
                self._move_toward_target(driver, driver.assigned_rider.pickup_location)
                
        elif driver.state == DriverState.WITH_PASSENGER:
            # Move toward dropoff location
            if driver.assigned_rider:
                self._move_toward_target(driver, driver.assigned_rider.dropoff_location)
                
        elif driver.state == DriverState.OFF_DUTY:
            # Move toward nearest off-duty cluster
            nearest_cluster = self.off_duty_dist.get_nearest_cluster(driver.location)
            if nearest_cluster:
                self._move_toward_target(driver, nearest_cluster.center)
    
    def _move_toward_target(self, driver, target):
        """Move driver toward a target location"""
        dx = np.sign(target[0] - driver.location[0])
        dy = np.sign(target[1] - driver.location[1])
        new_x = max(0, min(self.road_network.number_of_nodes()**0.5, driver.location[0] + dx))
        new_y = max(0, min(self.road_network.number_of_nodes()**0.5, driver.location[1] + dy))
        driver.location = (new_x, new_y)
    
    def _update_driver_states(self):
        """Update driver states based on simulation rules"""
        for driver in self.drivers:
            if driver.state == DriverState.CRUISING:
                # Chance to go off-duty or get assigned to a rider
                if random.random() < DRIVER_STATE_CHANGE_RATE:
                    driver.state = DriverState.OFF_DUTY
                else:
                    # Try to assign to nearest unmatched rider
                    self._try_assign_rider(driver)
                    
            elif driver.state == DriverState.ASSIGNED:
                # Check if reached pickup location
                if driver.assigned_rider and self._has_reached_location(driver, driver.assigned_rider.pickup_location):
                    driver.state = DriverState.WITH_PASSENGER
                    
            elif driver.state == DriverState.WITH_PASSENGER:
                # Check if reached dropoff location
                if driver.assigned_rider and self._has_reached_location(driver, driver.assigned_rider.dropoff_location):
                    driver.state = DriverState.CRUISING
                    driver.assigned_rider = None
                    
            elif driver.state == DriverState.OFF_DUTY:
                # Check if should return to cruising
                if random.random() < DRIVER_STATE_CHANGE_RATE:
                    driver.state = DriverState.CRUISING
    
    def _get_valid_neighbors(self, pos):
        """Get all valid neighboring positions"""
        grid_size = int(self.road_network.number_of_nodes() ** 0.5)
        neighbors = []
        for offset in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            new_x = pos[0] + offset[0]
            new_y = pos[1] + offset[1]
            # Apply periodic boundary conditions
            new_x = new_x % grid_size
            new_y = new_y % grid_size
            if self.road_network.has_edge(pos, (new_x, new_y)):
                neighbors.append((new_x, new_y))
        return neighbors
    
    def _update_cruising_drivers(self):
        """Update cruising drivers with oscillation prevention"""
        for driver in self.drivers:
            if driver.state == DriverState.CRUISING:
                # Use existing demand center logic
                closest_center = min(self.demand_centers, 
                    key=lambda x: self._get_periodic_distance(driver.location, x))
                self._update_driver_position(driver, closest_center)
    
    def _match_passengers_to_drivers(self):
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
                distance = np.linalg.norm(
                    np.array(driver.location) - np.array(rider.pickup_location)
                )
                travel_time = distance  # Assuming unit speed
                travel_times.append((driver, rider, travel_time))
        
        # Sort by travel time and match closest pairs
        travel_times.sort(key=lambda x: x[2])
        matched_drivers = set()
        matched_riders = set()
        
        for driver, rider, _ in travel_times:
            if (driver not in matched_drivers and rider not in matched_riders):
                driver.state = DriverState.ASSIGNED
                driver.assigned_rider = rider
                rider.assigned_driver = driver
                matched_drivers.add(driver)
                matched_riders.add(rider)
    
    def _try_assign_rider(self, driver):
        """Attempt to assign the nearest rider to the driver"""
        if driver.state != DriverState.CRUISING:
            return
        
        # Find the nearest unmatched rider
        nearest_rider = None
        min_distance = float('inf')
        
        for rider in self.riders:
            if rider.assigned_driver is None:
                distance = np.linalg.norm(np.array(driver.location) - np.array(rider.pickup_location))
                if distance < min_distance:
                    min_distance = distance
                    nearest_rider = rider
        
        # Assign the rider to the driver if found
        if nearest_rider:
            driver.state = DriverState.ASSIGNED
            driver.assigned_rider = nearest_rider
            nearest_rider.assigned_driver = driver
