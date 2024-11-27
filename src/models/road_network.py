import networkx as nx
import numpy as np

def create_sample_road_network(size: int = 10) -> nx.Graph:
    # Create a simple grid network
    G = nx.grid_2d_graph(size, size)
    
    # Add edge weights (distances)
    for (u, v) in G.edges():
        G[u][v]['length'] = np.sqrt(
            (u[0] - v[0])**2 + (u[1] - v[1])**2
        )
    
    return G
