import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from scipy.special import gammaln as loggamma

# Function to calculate Bayesian score component
def bayesian_score_component(M, α):
    p = np.sum(loggamma(α + M))
    p -= np.sum(loggamma(α))
    p += np.sum(loggamma(np.sum(α, axis=1)))
    p -= np.sum(loggamma(np.sum(α + M, axis=1)))
    return p

# Function to calculate the Bayesian score for the entire graph
def bayesian_score(vars, G, D):
    n = len(vars)
    M = statistics(vars, G, D)  # Get the contingency tables
    α = prior(vars, G)  # Get the priors
    return sum(bayesian_score_component(M[i], α[i]) for i in range(n))

# Function to calculate the contingency tables for each variable
def statistics(vars, G, D):
    n = len(vars)
    r = [vars[i]['r'] for i in range(n)]
    q = [np.prod([r[parent] for parent in G.predecessors(i)]) for i in range(n)]
    M = [np.zeros((int(q[i]), int(r[i]))) for i in range(n)]
    for o in D.T:
        for i in range(n):
            k = int(o[i])  # Ensure k is an integer
            parents = list(G.predecessors(i))
            if not parents:
                j = 1  # No parents case
            else:
                j = int(sub2ind([r[parent] for parent in parents], o[parents]))  # Ensure j is an integer
            
            if 0 <= j-1 < M[i].shape[0] and 0 <= k-1 < M[i].shape[1]:
                M[i][j-1, k-1] += 1
    return M

# Function to calculate the prior for each variable
def prior(vars, G):
    n = len(vars)
    r = [vars[i]['r'] for i in range(n)]
    q = [np.prod([r[parent] for parent in G.predecessors(i)]) for i in range(n)]
    return [np.ones((int(q[i]), int(r[i]))) for i in range(n)]

# Function to calculate index for parents' configurations
def sub2ind(siz, x):
    k = np.hstack([1, np.cumprod(siz[:-1])])
    return np.dot(k, (np.array(x) - 1)) + 1

# K2 search algorithm with cycle prevention
def k2_search(vars, D, node_order, max_parents):
    G = nx.DiGraph()
    n = len(node_order)
    
    # Ensure all nodes are added to the graph first
    for node in node_order:
        G.add_node(node)
        
    # Now perform the K2 search to add edges
    for node in node_order:
        parents = []
        for potential_parent in node_order:
            if potential_parent == node or len(parents) >= max_parents:
                continue
            
            # Add the edge temporarily to check for cycles
            G.add_edge(potential_parent, node)
            if nx.is_directed_acyclic_graph(G):
                parents.append(potential_parent)
            else:
                # If a cycle is created, remove the edge
                G.remove_edge(potential_parent, node)
        
        score = bayesian_score(vars, G, D)  # Calculate the score at each step
    
    # Check if the graph is acyclic
    if nx.is_directed_acyclic_graph(G):
        print("Graph is acyclic!")
    else:
        print("Graph contains cycles. Please check the graph!")
    
    return G, parents

# Function to save the graph in .gph format with variable names
def save_gph_file_with_variable_names(G, filename, vars):
    # Create a mapping from node index to variable names
    mapping = {i: vars[i]['name'] for i in G.nodes()}
    G = nx.relabel_nodes(G, mapping)
    
    with open(filename, 'w') as f:
        for edge in G.edges():
            f.write(f"{edge[0]},{edge[1]}\n")  # Write in parent,child format

# Function to display the graph with variable names
def display_graph_with_variable_names(G, dataset_name, vars):
    # Create a mapping from node index to variable names
    mapping = {i: vars[i]['name'] for i in G.nodes()}
    G = nx.relabel_nodes(G, mapping)
    
    # Draw and display the graph
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G)  # Positioning of the nodes
    nx.draw(G, pos, with_labels=True, node_color='lightblue', font_weight='bold', node_size=3000)
    plt.title(f"Bayesian Network for {dataset_name}")
    plt.show()

# Function to run experiments with different max_parents and node_order
def experiment_with_parameters(vars, D, dataset_name):
    max_parents_values = [2, 4, 5]  # List of max_parents to experiment with
    node_order_values = [
        [0, 1, 2, 3, 4, 5, 6, 7],  # Natural order
        [7, 6, 5, 4, 3, 2, 1, 0],  # Reversed order
        [3, 1, 4, 0, 2, 6, 5, 7]   # Random order
    ]

    best_score = float('-inf')  # Initialize best score
    best_graph = None  # Initialize best graph
    
    for max_parents in max_parents_values:
        for node_order in node_order_values:
            print(f"Running K2 search with max_parents={max_parents} and node_order={node_order}")
            G, parents = k2_search(vars, D, node_order, max_parents)

            # Calculate Bayesian score
            score = bayesian_score(vars, G, D)
            print(f"Bayesian Score: {score}")

            # If this score is better than the best score, save this graph
            if score > best_score:
                best_score = score
                best_graph = G

    # Save the best graph for this dataset in .gph format with variable names
    save_gph_file_with_variable_names(best_graph, f"/Users/macbookpro/Desktop/funmii/{dataset_name}_best_graph.gph", vars)
    print(f"Best graph saved for {dataset_name} with Bayesian Score: {best_score}\n")

    # Display the best graph with variable names
    display_graph_with_variable_names(best_graph, dataset_name, vars)

# Main function to run the Bayesian Network
def main():
    # Load the dataset (modify the path for medium or large datasets)
    data = pd.read_csv('/Users/macbookpro/Desktop/funmii/small.csv')
    vars = [
        {"name": "age", "r": 3},
        {"name": "portembarked", "r": 3},
        {"name": "fare", "r": 3},
        {"name": "numparentschildren", "r": 3},
        {"name": "passengerclass", "r": 3},
        {"name": "sex", "r": 2},
        {"name": "numsiblings", "r": 3},
        {"name": "survived", "r": 2},
    ]
    
    D = data.values.T  # Transpose data

    # Run the experiments with different max_parents and node_order for small dataset
    experiment_with_parameters(vars, D, "small")

if __name__ == "__main__":
    main()
