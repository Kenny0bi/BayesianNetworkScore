import numpy as np
import pandas as pd
import networkx as nx
from scipy.special import gammaln as loggamma
import matplotlib.pyplot as plt
import os

# Set a random seed for reproducibility
np.random.seed(42)

# Function to calculate Bayesian score component
def bayesian_score_component(M, α):
    p = np.sum(loggamma(α + M))
    p -= np.sum(loggamma(α))
    p += np.sum(loggamma(np.sum(α, axis=1)))
    p -= np.sum(loggamma(np.sum(α + M, axis=1)))
    return p

# Function to calculate the Bayesian score for the entire graph
def bayesian_score(vars, G, D, ess=1.0):
    n = len(vars)
    M = statistics(vars, G, D)  # Get the contingency tables
    α = prior(vars, G, ess)  # Get the priors with ESS
    return sum(bayesian_score_component(M[i], α[i]) for i in range(n))

# Function to calculate the contingency tables for each variable
def statistics(vars, G, D):
    n = len(vars)
    r = [vars[i]['r'] for i in range(n)]
    q = [np.prod([r[parent] for parent in G.predecessors(i)]) for i in range(n)]
    M = [np.zeros((int(q[i]), int(r[i]))) for i in range(n)]
    
    for o in D.T:
        for i in range(n):
            k = o[i]
            
            # Check if the value of k is out of bounds or NaN
            if pd.isna(k) or k > r[i] or k < 1:
                raise ValueError(f"Value {k} for variable {vars[i]['name']} is out of bounds or NaN. Expected range: 1 to {r[i]}")
            
            parents = list(G.predecessors(i))
            if not parents:
                j = 1
            else:
                j = int(sub2ind([r[parent] for parent in parents], o[parents]))  # Ensure j is an integer
            
            # Validate j and k before accessing the matrix
            if j-1 >= M[i].shape[0] or k-1 >= M[i].shape[1] or j-1 < 0 or k-1 < 0:
                raise IndexError(f"Index j={j-1} or k={k-1} out of bounds for M[i] with shape {M[i].shape}")

            # Update the contingency table M for the variable
            M[i][j-1, k-1] += 1
    return M

# Function to calculate the prior for each variable, incorporating ESS (Equivalent Sample Size)
def prior(vars, G, ess=1.0):
    n = len(vars)
    r = [vars[i]['r'] for i in range(n)]
    q = [np.prod([r[parent] for parent in G.predecessors(i)]) for i in range(n)]
    return [ess * np.ones((int(q[i]), int(r[i]))) for i in range(n)]

# Function to calculate index for parents' configurations
def sub2ind(siz, x):
    k = np.hstack([1, np.cumprod(siz[:-1])])
    return int(np.dot(k, (np.array(x) - 1)) + 1)

# K2 search algorithm with acyclic guarantee
def k2_search(vars, D, node_order, max_parents, ess=1.0):
    G = nx.DiGraph()
    n = len(node_order)
    
    # Ensure all nodes are added to the graph first
    for node in node_order:
        G.add_node(node)
        
    # Now perform the K2 search to add edges while ensuring acyclicity
    for node in node_order:
        parents = []
        for potential_parent in node_order:
            if potential_parent == node or len(parents) >= max_parents:
                continue
            G.add_edge(potential_parent, node)
            
            # Check if adding this edge creates a cycle
            if nx.is_directed_acyclic_graph(G):
                parents.append(potential_parent)
            else:
                G.remove_edge(potential_parent, node)
                
        score = bayesian_score(vars, G, D, ess)  # Calculate the score at each step with ESS
    return G, parents

# Function to display and save the graph in .gph and .gml files
def display_and_save_graph(G, graph_name_gph, graph_name_gml, variable_names):
    # Rename nodes in the graph to variable names
    mapping = {i: variable_names[i] for i in G.nodes()}
    G = nx.relabel_nodes(G, mapping)
    
    # Draw and display the graph in a pop-up window
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G, seed=42)  # Fixed seed for reproducibility
    nx.draw(G, pos, with_labels=True, node_color='lightblue', font_weight='bold', node_size=3000, font_size=10)
    plt.title("Bayesian Network Graph")
    plt.show()

    # Save the graph as a .gph file (Graphviz format)
    with open(graph_name_gph, 'w') as f:
        for edge in G.edges():
            f.write(f"{edge[0]},{edge[1]}\n")  # Write in parent,child format

    print(f"Graph saved as {graph_name_gph}")
    
    # Save the graph as a .gml file
    nx.write_gml(G, graph_name_gml)
    print(f"Graph saved as {graph_name_gml}")

# Experiment function to test various ESS, max_parents, and node orders
def experiment_with_parameters(vars, D, variable_names, dataset_path):
    best_score = float('-inf')  # Initialize best score
    best_graph = None  # Initialize best graph

    # Parameter values to experiment with
    ess_values = [0.5, 1.0, 2.0, 5.0]
    max_parents_values = [2, 3, 4]
    node_order_values = [
        list(range(len(vars))),  # Natural order
        list(reversed(range(len(vars)))),  # Reversed order
        np.random.permutation(len(vars)).tolist()  # Random order
    ]

    for ess in ess_values:
        for max_parents in max_parents_values:
            for node_order in node_order_values:
                print(f"Running K2 search with ESS={ess}, max_parents={max_parents}, and node_order={node_order}")
                G, parents = k2_search(vars, D, node_order, max_parents, ess)

                # Calculate Bayesian score
                score = bayesian_score(vars, G, D, ess)
                print(f"Bayesian Score: {score}")

                # If this score is better than the best score, save this graph
                if score > best_score:
                    best_score = score
                    best_graph = G

    # Save the best graph for this dataset
    graph_name_gph = os.path.join(os.path.dirname(dataset_path), "large_best_graph.gph")
    graph_name_gml = os.path.join(os.path.dirname(dataset_path), "large_best_graph.gml")

    display_and_save_graph(best_graph, graph_name_gph, graph_name_gml, variable_names)

    print(f"Best Bayesian Score: {best_score}")

# Main function to run the Bayesian Network
def main():
    # Load the dataset (large dataset example)
    dataset_path = '/Users/macbookpro/Desktop/funmii/large.csv'
    data = pd.read_csv(dataset_path)

    # Handle missing values by filling with mode (most frequent value)
    data.fillna(data.mode().iloc[0], inplace=True)

    # Variables with unique values (based on provided details)
    vars = [
        {"name": "DG", "r": 3}, {"name": "YP", "r": 3}, {"name": "ME", "r": 4}, {"name": "OF", "r": 2},
        {"name": "ZX", "r": 2}, {"name": "QZ", "r": 4}, {"name": "RE", "r": 3}, {"name": "CA", "r": 2},
        {"name": "WF", "r": 2}, {"name": "WW", "r": 2}, {"name": "SD", "r": 3}, {"name": "EY", "r": 2},
        {"name": "TU", "r": 3}, {"name": "IB", "r": 3}, {"name": "IM", "r": 3}, {"name": "BQ", "r": 4},
        {"name": "VU", "r": 2}, {"name": "MY", "r": 2}, {"name": "RZ", "r": 3}, {"name": "JI", "r": 2},
        {"name": "FR", "r": 3}, {"name": "UO", "r": 2}, {"name": "XT", "r": 3}, {"name": "UQ", "r": 4},
        {"name": "HH", "r": 3}, {"name": "PM", "r": 4}, {"name": "QS", "r": 4}, {"name": "TV", "r": 4},
        {"name": "CZ", "r": 3}, {"name": "YQ", "r": 2}, {"name": "UN", "r": 2}, {"name": "US", "r": 4},
        {"name": "YR", "r": 3}, {"name": "EV", "r": 4}, {"name": "OG", "r": 4}, {"name": "KE", "r": 3},
        {"name": "ND", "r": 2}, {"name": "YH", "r": 2}, {"name": "XQ", "r": 2}, {"name": "FZ", "r": 3},
        {"name": "GC", "r": 3}, {"name": "ZL", "r": 4}, {"name": "NX", "r": 3}, {"name": "BX", "r": 2},
        {"name": "MU", "r": 3}, {"name": "GS", "r": 2}, {"name": "DW", "r": 2}, {"name": "JP", "r": 2},
        {"name": "UC", "r": 4}, {"name": "WN", "r": 2}
    ]
    
    D = data.values.T  # Transpose data

    # Define variable names for the graph
    variable_names = [
        "DG", "YP", "ME", "OF", "ZX", "QZ", "RE", "CA", "WF", "WW", "SD", "EY", "TU", "IB", "IM", "BQ", "VU", "MY", "RZ", "JI", "FR",
        "UO", "XT", "UQ", "HH", "PM", "QS", "TV", "CZ", "YQ", "UN", "US", "YR", "EV", "OG", "KE", "ND", "YH", "XQ", "FZ", "GC", "ZL",
        "NX", "BX", "MU", "GS", "DW", "JP", "UC", "WN"
    ]

    # Run experiments to find the best Bayesian score
    experiment_with_parameters(vars, D, variable_names, dataset_path)

if __name__ == "__main__":
    main()
