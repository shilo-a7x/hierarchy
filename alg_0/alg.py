import os
import pickle
import networkx as nx
from networkx.algorithms.community import louvain_communities
import argparse
import json
import numpy as np


# Centrality-based score functions
def betweenness_centrality_score(graph):
    """Calculate betweenness centrality for each node."""
    return nx.betweenness_centrality(graph.to_undirected())


def pagerank_score(graph):
    """Calculate PageRank for each node."""
    return nx.pagerank(graph)


def closeness_centrality_score(graph):
    """Calculate closeness centrality for each node."""
    return nx.closeness_centrality(graph)


def in_degree_centrality_score(graph):
    """Calculate in-degree centrality for each node."""
    if not graph.is_directed():
        return degree_centrality_score(graph)
    return nx.in_degree_centrality(graph)


def out_degree_centrality_score(graph):
    """Calculate out-degree centrality for each node."""
    if not graph.is_directed():
        return degree_centrality_score(graph)
    return nx.out_degree_centrality(graph)


def hits_score(graph):
    """Calculate HITS (Hyperlink-Induced Topic Search) scores for each node."""
    hubs, authorities = nx.hits(graph)
    return hubs  # Return hub scores as the score


def katz_score(graph, alpha=0.005, beta=1.0):
    """Calculate Katz centrality for each node."""
    return nx.katz_centrality(graph, alpha=alpha, beta=beta)


def triangle_score(graph):
    """Calculate triangle count for each node."""
    return nx.triangles(graph.to_undirected())


def clustering_coefficient_score(graph):
    """Calculate clustering coefficient for each node."""
    return nx.clustering(graph.to_undirected())


def degree_centrality_score(graph):
    """Calculate degree centrality for each node."""
    return nx.degree_centrality(graph)


def local_efficiency_score(graph):
    """Calculate local efficiency for each node."""
    undirected_graph = graph.to_undirected()
    return {
        node: nx.local_efficiency(
            undirected_graph.subgraph([node] + list(undirected_graph.neighbors(node)))
        )
        for node in undirected_graph.nodes()
    }


def core_number_score(graph):
    """Calculate core number (k-core decomposition) for each node."""
    return nx.core_number(graph)


# Spectral-based score functions
def spectral_centrality_score(graph):
    """Calculate the spectral centrality using the adjacency matrix."""
    adj_matrix = nx.adjacency_matrix(graph).todense()
    eigvals, eigvecs = np.linalg.eig(adj_matrix)
    spectral_centrality = np.abs(eigvecs[:, np.argmax(eigvals)])

    # Return as a dictionary with node labels as keys
    return {node: spectral_centrality[i] for i, node in enumerate(graph.nodes)}


def fiedler_vector_score(graph):
    """Calculate the Fiedler vector (algebraic connectivity) using the Laplacian matrix."""
    laplacian_matrix = nx.laplacian_matrix(graph.to_undirected()).todense()
    eigvals, eigvecs = np.linalg.eig(laplacian_matrix)
    fiedler_vector = eigvecs[
        :, np.argsort(eigvals)[1]
    ]  # Second smallest eigenvalue's vector

    # Return as a dictionary with node labels as keys
    return {node: fiedler_vector[i].real for i, node in enumerate(graph.nodes)}


def laplacian_eigenmap_score(graph, k=5):
    """Calculate the Laplacian Eigenmap by embedding the graph into k dimensions."""
    laplacian_matrix = nx.laplacian_matrix(graph.to_undirected()).todense()
    eigvals, eigvecs = np.linalg.eig(laplacian_matrix)
    embedding = eigvecs[
        :, np.argsort(eigvals)[1 : k + 1]
    ]  # Use k smallest eigenvectors

    # Return the first component for simplicity
    return {node: embedding[i, 0].real for i, node in enumerate(graph.nodes)}


def normalized_laplacian_score(graph):
    """Calculate the normalized Laplacian centrality."""
    normalized_laplacian_matrix = nx.normalized_laplacian_matrix(
        graph.to_undirected()
    ).todense()
    eigvals, eigvecs = np.linalg.eig(normalized_laplacian_matrix)
    normalized_laplacian = np.abs(eigvecs[:, np.argmax(eigvals)])

    # Return as a dictionary with node labels as keys
    return {node: normalized_laplacian[i] for i, node in enumerate(graph.nodes)}


def spectral_radius_score(graph):
    """Calculate the spectral radius (largest eigenvalue of the adjacency matrix)."""
    adj_matrix = nx.adjacency_matrix(graph).todense()
    eigvals = np.linalg.eigvals(adj_matrix)
    spectral_radius = np.max(np.abs(eigvals))

    # Return as a dictionary with the spectral radius assigned to all nodes
    return {node: spectral_radius for node in graph.nodes}


def spectral_clustering_score(graph, k=2):
    """Perform spectral clustering and return the clustering results for each node."""
    laplacian_matrix = nx.laplacian_matrix(graph.to_undirected()).todense()
    eigvals, eigvecs = np.linalg.eig(laplacian_matrix)
    clusters = np.argsort(eigvecs[:, :k], axis=1).flatten()

    # Return as a dictionary with node labels as keys
    return {node: clusters[i] for i, node in enumerate(graph.nodes)}


# Add additional score functions here if necessary


# Function to load the configuration from a JSON file
def load_config(config_path):
    if not os.path.exists(config_path):
        return None
    with open(config_path, "r") as f:
        config = json.load(f)
    return config


# Load the score function mapping and the scores to run from the config
def get_score_functions(config=None):
    # Default score functions mapping
    score_functions = {
        "betweenness": betweenness_centrality_score,
        "pagerank": pagerank_score,
        "closeness": closeness_centrality_score,
    }

    # If a config is provided, update score functions
    if config and "score_functions" in config:
        for score_name, score_func in config["score_functions"].items():
            if score_func in globals():
                score_functions[score_name] = globals()[score_func]
            else:
                print(f"Warning: '{score_func}' function not found.")
                raise NotImplementedError

    return score_functions


# Modify this function to dynamically use the correct score function
def graph_nodes_score(graph, score_type, score_functions):
    """Return the node scores based on the selected score function."""
    if score_type not in score_functions:
        raise ValueError(f"Unknown score type: {score_type}")
    score_function = score_functions[score_type]
    return score_function(graph)


def dag_from_score(graph, score) -> nx.DiGraph:
    """Create a Directed Acyclic Graph (DAG) from score."""
    dag = nx.DiGraph()
    dag.add_nodes_from(graph.nodes)
    for u, v in graph.edges:
        if score[u] > score[v]:
            dag.add_edge(u, v)
        elif score[u] < score[v]:
            dag.add_edge(v, u)
    return dag


def tree_from_subdags(subdags, score) -> nx.DiGraph:
    subdags_sorted = []
    for s in subdags:
        head_nodes = [v for v in s if s.in_degree(v) == 0]
        if not head_nodes:
            raise ValueError("Subdag has no head nodes")
        heads = sorted(head_nodes, key=lambda v: score[v], reverse=True)
        leaves = sorted(
            [v for v in s if s.out_degree(v) == 0], key=lambda v: score[v], reverse=True
        )
        subdags_sorted.append(
            {
                "subgraph": s,
                "heads": heads,
                "score": score[heads[0]],
                "leaves": leaves,
            }
        )
    subdags_sorted.sort(key=lambda x: x["score"], reverse=True)

    combined_graph = nx.DiGraph()

    top_community = subdags_sorted[0]
    top_subgraph = top_community["subgraph"]
    top_head = top_community["heads"][0]
    for other_head in top_community["heads"][1:]:
        combined_graph.add_edge(top_head, other_head)

    combined_graph.add_edges_from(top_subgraph.edges)

    prev_leaves = top_community["leaves"]
    for i in range(1, len(subdags_sorted)):
        current_community = subdags_sorted[i]
        current_subgraph = current_community["subgraph"]
        current_heads = current_community["heads"]

        for idx, head in enumerate(current_heads):
            if idx < len(prev_leaves):
                combined_graph.add_edge(prev_leaves[idx], head)
            else:
                combined_graph.add_edge(prev_leaves[-1], head)

        combined_graph.add_edges_from(current_subgraph.edges)

        prev_leaves = current_community["leaves"]

    root = top_head
    tree = nx.bfs_tree(combined_graph, source=root)
    return tree


def alg(graph: nx.Graph, score_type, score_functions):
    """Generate tree from graph using a specific score function."""
    graph_score = graph_nodes_score(graph, score_type, score_functions)
    dag = dag_from_score(graph, graph_score)
    # communities = louvain_communities(dag.to_undirected(), threshold=0.8)
    # community_subgraphs = [dag.subgraph(community).copy() for community in communities]
    # hierarchy_tree = tree_from_subdags(community_subgraphs, graph_score)
    # print(nx.is_tree(hierarchy_tree))
    # return hierarchy_tree
    return dag


def load_graph(file_path):
    with open(file_path, "rb") as f:
        graph = pickle.load(f)
    return graph


def save_graph(graph, file_path):
    with open(file_path, "wb") as f:
        pickle.dump(graph, f)
    print(f"Saved tree to {file_path}")


def process_graph(input_file, output_file, score_type, score_functions):
    """Process the graph with a specified score function."""
    graph = load_graph(input_file)
    result_tree = alg(graph, score_type, score_functions)
    save_graph(result_tree, output_file)


def main():
    parser = argparse.ArgumentParser(description="Process a graph and output a tree")

    parser.add_argument(
        "input_file", type=str, help="Path to the input graph pickle file"
    )
    parser.add_argument(
        "output_file",
        type=str,
        help="Path to save the output tree pickle file",
    )
    parser.add_argument(
        "--score",
        type=str,
        default="betweenness",  # Default to 'betweenness' if not provided
        help="Specify the score function to use (e.g., 'betweenness', 'pagerank', 'closeness', etc.)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.json",
        help="Path to the configuration JSON file (default: 'config.json')",
    )

    args = parser.parse_args()

    # Load configuration from file, fallback to defaults if not found
    config = load_config(args.config)
    score_functions = get_score_functions(config)

    # Process the graph using the specified score function
    process_graph(args.input_file, args.output_file, args.score, score_functions)


if __name__ == "__main__":
    main()
