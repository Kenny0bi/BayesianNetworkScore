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
    M = statistics(vars, G, D)
    α = prior(vars, G, ess)
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
            if pd.isna(k) or k > r[i] or k < 1:
                raise ValueError(f"Value {k} for variable {vars[i]['name']} is out of bounds or NaN.")
            parents = list(G.predecessors(i))
            j = 1 if not parents else sub2ind([r[parent] for parent in parents], o[parents])
            M[i][j-1, int(k)-1] += 1
    return M

# Function to calculate the prior for each variable
def prior(vars, G, ess=1.0):
    n = len(vars)
    r = [vars[i]['r'] for i in range(n)]
    q = [np.prod([r[parent] for parent in G.predecessors(i)]) for i in range(n)]
    return [ess * np.ones((int(q[i]), int(r[i]))) for i in range(n)]

# Function to calculate index for parents' configurations
def sub2ind(siz, x):
    k = np.hstack([1, np.cumprod(siz[:-1])])
    return int(np.dot(k, (np.array(x) - 1)) + 1)

# K2 search algorithm to ensure no cycles
def k2_search(vars, D, node_order, max_parents, ess=1.0):
    G = nx.DiGraph()
    n = len(node_order)
    
    for node in node_order:
        G.add_node(node)
        
    for node in node_order:
        parents = []
        for potential_parent in node_order:
            if potential_parent == node or len(parents) >= max_parents:
                continue
            G.add_edge(potential_parent, node)
            if not nx.is_directed_acyclic_graph(G):  # Ensure the graph remains acyclic
                G.remove_edge(potential_parent, node)  # Remove the edge if it creates a cycle
            else:
                parents.append(potential_parent)
        score = bayesian_score(vars, G, D, ess)
    return G, parents

# Function to display and save the graph
def display_and_save_graph(G, graph_name_gph, graph_name_gml, variable_names):
    mapping = {i: variable_names[i] for i in G.nodes()}
    G = nx.relabel_nodes(G, mapping)
    
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', font_weight='bold', node_size=3000, font_size=10)
    plt.title("Bayesian Network Graph")
    plt.show()

    with open(graph_name_gph, 'w') as f:
        for edge in G.edges():
            f.write(f"{edge[0]},{edge[1]}\n")
    print(f"Graph saved as {graph_name_gph}")

    nx.write_gml(G, graph_name_gml)
    print(f"Graph saved as {graph_name_gml}")

# Main function to run the Bayesian Network
def main():
    dataset_path = '/Users/macbookpro/Desktop/funmii/medium.csv'
    data = pd.read_csv(dataset_path)

    alcohol_bins = [1, 3, 5, 6, 7, 8]
    fixedacidity_bins = [1, 2, 3, 4, 5, 6]

    data['alcohol'] = pd.cut(data['alcohol'], bins=alcohol_bins, labels=[1, 2, 3, 4, 5], include_lowest=True)
    data['fixedacidity'] = pd.cut(data['fixedacidity'], bins=fixedacidity_bins, labels=[1, 2, 3, 4, 5], include_lowest=True)

    data['alcohol'] = data['alcohol'].fillna(data['alcohol'].mode()[0])
    data['fixedacidity'] = data['fixedacidity'].fillna(data['fixedacidity'].mode()[0])

    vars = [
        {"name": "color", "r": 2}, {"name": "fixedacidity", "r": 5}, {"name": "volatileacidity", "r": 5},
        {"name": "citricacid", "r": 5}, {"name": "residualsugar", "r": 5}, {"name": "chlorides", "r": 5},
        {"name": "freesulfurdioxide", "r": 5}, {"name": "totalsulfurdioxide", "r": 5}, {"name": "density", "r": 5},
        {"name": "pH", "r": 5}, {"name": "sulphates", "r": 5}, {"name": "alcohol", "r": 5}, {"name": "quality", "r": 8},
    ]
    
    D = data.values.T
    variable_names = ["color", "fixedacidity", "volatileacidity", "citricacid", "residualsugar", "chlorides", 
                      "freesulfurdioxide", "totalsulfurdioxide", "density", "pH", "sulphates", "alcohol", "quality"]

    node_order = [3, 1, 4, 0, 2, 6, 5, 7, 8, 9, 10, 11, 12]
    max_parents = 4
    ess = 1.0

    G, parents = k2_search(vars, D, node_order, max_parents, ess)
    score = bayesian_score(vars, G, D, ess)
    print(f"Bayesian Score: {score}")

    graph_name_gph = os.path.join(os.path.dirname(dataset_path), "medium_graph.gph")
    graph_name_gml = os.path.join(os.path.dirname(dataset_path), "medium_graph.gml")
    
    display_and_save_graph(G, graph_name_gph, graph_name_gml, variable_names)

if __name__ == "__main__":
    main()
