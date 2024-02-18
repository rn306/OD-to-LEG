import numpy as np
import networkx as nx
from sklearn.metrics import pairwise_distances

def generate_graph(node_coord, node_population, freqmin):
    
    n_nodes = len(node_coord)
    population_matrix = node_population.reshape(-1, 1)@node_population.reshape(1, -1)

    # Get graph properties
    distance_matrix = pairwise_distances(node_coord)
    np.fill_diagonal(distance_matrix, 0.)
    
    od_matrix = np.divide(population_matrix, distance_matrix**2,
                          out = np.zeros_like(population_matrix), where = distance_matrix>=0.0001)
    
    # Cost parameters. Assume a fixed cost per seat of 1 and all modules have 10 seats
    # Only FreqMin is a variable since this is the driver of sparsificaton
    freqmin_matrix = np.where(distance_matrix == 0, 0, freqmin)
    targetcapa_matrix = np.where(distance_matrix == 0, 0, 10)
    CPS_matrix = np.where(distance_matrix == 0, 0, 1)

    # Create graph
    dt = np.dtype([('OD_Pax', '<f8'), ('Distance', '<f8'), ('FreqMin', '<f8'), ('TargetCapa', '<f8'), ('CPS', '<f8')])

    full_array = np.array((list(zip(od_matrix.flatten(),
                                    distance_matrix.flatten(),
                                    freqmin_matrix.flatten(),
                                    targetcapa_matrix.flatten(),
                                    CPS_matrix.flatten()))), dtype = dt).reshape(n_nodes, n_nodes)

    G_OD_full = nx.from_numpy_matrix(full_array)
    
    return G_OD_full