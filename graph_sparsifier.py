import numpy as np
import networkx as nx
import cvxpy as cp

def graph_sparsification(G_OD_full):

    # Get incidence matrix
    B = nx.incidence_matrix(G_OD_full, oriented = True).toarray().T
    m, n = B.shape
    
    # Get the edge properties with correct order
    edges = G_OD_full.edges()
    od_array = np.array([G_OD_full[u][v]['OD_Pax'] for u, v in edges])
    distance_array = np.array([G_OD_full[u][v]['Distance'] for u, v in edges])
    freqmin_array = np.array([G_OD_full[u][v]['FreqMin'] for u, v in edges])
    targetcapa_array = np.array([G_OD_full[u][v]['TargetCapa'] for u, v in edges])
    cps_array = np.array([G_OD_full[u][v]['CPS'] for u, v in edges])
    
    # Initialize variables
    R =  cp.Variable((m,m)) 
    R_abs = cp.Variable((m,m))
    L = cp.Variable((m,))
    cost_L = cp.Variable((m,))
    z_indicator = cp.Variable((m,), boolean=True)

    # Prepare constraints
    M = 100000
    constraints = []

    constraints.append(R_abs >= 0)
    constraints.append(L >= 0)
    constraints.append(cost_L >= 0)

    constraints.append(B.T@R == B.T@np.diag(od_array))
    constraints.append(R_abs >= R); constraints.append(R_abs >= -R)
    constraints.append(cp.sum(R_abs, axis = 1) == L)

    constraints.append(L >= z_indicator*(1/M))
    constraints.append(L <= M*z_indicator)
    constraints.append(cost_L >= distance_array*cps_array*freqmin_array*targetcapa_array - M*(1 - z_indicator))
    constraints.append(cost_L >= cp.multiply(distance_array*cps_array, L)-M*(1-z_indicator))

    obj = cp.Minimize(sum(cost_L))
    problem = cp.Problem(obj, constraints=constraints)

    problem.solve(solver = cp.GLPK_MI)

    # Report
    result = np.abs(L.value)
    
    return result