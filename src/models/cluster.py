import numpy as np
from scipy.stats import multivariate_normal
import random
from typing import Tuple, List

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
    
    def get_nearest_cluster(self, location):
        """Find the nearest cluster to a given location"""
        nearest_cluster = None
        min_distance = float('inf')
        
        for cluster in self.clusters:
            distance = np.linalg.norm(np.array(location) - np.array(cluster.center))
            if distance < min_distance:
                min_distance = distance
                nearest_cluster = cluster
        
        return nearest_cluster
