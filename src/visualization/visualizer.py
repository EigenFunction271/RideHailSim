import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as patches
import numpy as np
import time

from models.driver import DriverState
from simulation.simulation import RideHailSimulation

class RideHailVisualization:
    def __init__(self, simulation: RideHailSimulation, update_interval: int = 50):
        # [Previous implementation of RideHailVisualization]
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
        
        # Initialize rider plot
        self.rider_plot = self.map_ax.scatter([], [], c='black', marker='x', s=100, label='waiting riders')
        
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
            Z[i] = self.sim.off_duty_dist.get_density(pos)
        
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
    
    def setup_rider_heatmap(self):
        """Setup the rider heatmap visualization"""
        x = np.linspace(0, self.grid_size, 50)
        y = np.linspace(0, self.grid_size, 50)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)
        self.rider_heatmap = self.ax.pcolormesh(X, Y, Z, cmap='YlOrRd', alpha=0.3, shading='auto')
    
    def update(self, frame):
        """Update visualization for the current frame"""
        self.sim.step(dt=0.1)
        
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
        current_time = time.time() - self.start_time
        self.times.append(current_time)
        
        # Count drivers in each state
        with_passenger = len([d for d in self.sim.drivers if d.state == DriverState.WITH_PASSENGER])
        cruising = len([d for d in self.sim.drivers if d.state == DriverState.CRUISING])
        off_duty = len([d for d in self.sim.drivers if d.state == DriverState.OFF_DUTY])
        waiting_riders = len([r for r in self.sim.riders if not r.assigned_driver])
        
        # Update statistics tracking
        self.with_passenger_counts.append(with_passenger)
        self.rider_counts.append(waiting_riders)
        self.cruising_counts.append(cruising)
        self.off_duty_counts.append(off_duty)
        
        # Keep only last 60 seconds of data
        while self.times and self.times[0] < current_time - 60:
            self.times.pop(0)
            self.with_passenger_counts.pop(0)
            self.rider_counts.pop(0)
            self.cruising_counts.pop(0)
            self.off_duty_counts.pop(0)
        
        # Update the plot lines
        self.with_passenger_line.set_data(self.times, self.with_passenger_counts)
        self.rider_line.set_data(self.times, self.rider_counts)
        self.cruising_line.set_data(self.times, self.cruising_counts)
        self.off_duty_line.set_data(self.times, self.off_duty_counts)
        
        # Adjust axes
        if self.times:
            self.stats_ax.set_xlim(max(0, current_time - 60), current_time)
            max_count = max(
                max(self.with_passenger_counts + [0]),
                max(self.rider_counts + [0]),
                max(self.cruising_counts + [0]),
                max(self.off_duty_counts + [0])
            )
            self.stats_ax.set_ylim(0, max_count * 1.1)
        
        return (tuple(self.driver_plots.values()) + 
                (self.rider_plot,) +
                (self.with_passenger_line, self.rider_line, 
                 self.cruising_line, self.off_duty_line))
    
    def show(self):
        plt.show()