import pickle
import argparse
import random
import networkx as nx
import leidenalg as la
import igraph as ig
import numpy as np
from networkx.algorithms.tree.branchings import Edmonds


def convert_to_igraph(graph: nx.Graph):
    node_mapping = {node: idx for idx, node in enumerate(graph.nodes())}
    reverse_mapping = {idx: node for node, idx in node_mapping.items()}

    ig_graph = ig.Graph(directed=graph.is_directed())
    ig_graph.add_vertices(len(node_mapping))
    ig_graph.add_edges([(node_mapping[u], node_mapping[v]) for u, v in graph.edges()])

    return ig_graph, reverse_mapping


def perform_leiden(graph: nx.Graph):
    ig_graph, reverse_mapping = convert_to_igraph(graph)
    partition = la.find_partition(ig_graph, partition_type=la.ModularityVertexPartition)

    communities = {}
    for comm, nodes in enumerate(partition):
        for node in nodes:
            communities[reverse_mapping[node]] = comm

    return communities


def get_random_spanning_tree(G: nx.DiGraph):
    valid_roots = [node for node in G.nodes() if G.out_degree(node) > 0]
    if not valid_roots:
        raise ValueError("Graph has no valid root with outgoing edges")

    root = random.choice(valid_roots)

    tree = nx.DiGraph()
    tree.add_node(root)
    edges = list(G.edges(data=True))
    random.shuffle(edges)

    while len(tree.nodes) < len(G.nodes):
        for u, v, data in edges:
            if u in tree.nodes and v not in tree.nodes:
                tree.add_edge(u, v, weight=data.get("weight", 1))
                if len(tree.nodes) == len(G.nodes):
                    return tree
    return tree


def compute_initial_loss(
    tree: nx.DiGraph, true_tree: nx.DiGraph, communities, lambda_c=1.0
):
    weight_loss = sum(tree[u][v]["weight"] for u, v in tree.edges())

    true_out_degrees = np.array(list(dict(true_tree.out_degree()).values()))
    current_out_degrees = np.array(list(dict(tree.out_degree()).values()))

    degree_loss = np.mean((true_out_degrees - current_out_degrees) ** 2)

    community_penalty = sum(
        lambda_c for u, v in tree.edges() if communities[u] != communities[v]
    )

    return weight_loss, degree_loss, community_penalty, current_out_degrees


def compute_delta_loss(
    true_out_degrees,
    current_out_degrees,
    u_old,
    u_new,
    num_edges,
    old_edge,
    new_edge,
    communities,
    lambda_c=1.0,
):
    old_mse = (
        (true_out_degrees[u_old] - current_out_degrees[u_old]) ** 2
        + (true_out_degrees[u_new] - current_out_degrees[u_new]) ** 2
    ) / num_edges

    current_out_degrees[u_old] -= 1
    current_out_degrees[u_new] += 1

    new_mse = (
        (true_out_degrees[u_old] - current_out_degrees[u_old]) ** 2
        + (true_out_degrees[u_new] - current_out_degrees[u_new]) ** 2
    ) / num_edges

    delta_degree_loss = new_mse - old_mse

    old_penalty = (
        lambda_c if communities[old_edge[0]] != communities[old_edge[1]] else 0
    )
    new_penalty = (
        lambda_c if communities[new_edge[0]] != communities[new_edge[1]] else 0
    )
    delta_community_penalty = new_penalty - old_penalty

    return delta_degree_loss, delta_community_penalty


def modify_tree(tree: nx.DiGraph, original_graph: nx.DiGraph):
    edges = list(tree.edges())
    all_possible_edges = set(original_graph.edges()) - set(edges)
    if not edges or not all_possible_edges:
        return None, None
    old_edge = random.choice(edges)
    new_edge = random.choice(list(all_possible_edges))
    tree.remove_edge(*old_edge)
    tree.add_edge(*new_edge)
    if nx.is_arborescence(tree):
        return old_edge, new_edge
    tree.remove_edge(*new_edge)
    tree.add_edge(*old_edge)
    return None, None


def simulated_annealing(
    graph: nx.DiGraph,
    true_tree: nx.DiGraph,
    communities,
    lambda_c=1.0,
    max_iter=10000,
    initial_temp=100,
    cooling_rate=0.99,
):
    print("Starting simulated annealing")
    tree = Edmonds(graph).find_optimum(attr="weight", kind="min")
    # tree = get_random_spanning_tree(graph)
    best_tree = tree.copy()

    weight_loss, degree_loss, community_penalty, current_out_degrees = (
        compute_initial_loss(tree, true_tree, communities, lambda_c)
    )
    current_loss = weight_loss + degree_loss + community_penalty
    best_loss = current_loss

    temperature = initial_temp
    num_edges = len(tree.edges())
    true_out_degrees = dict(true_tree.out_degree())

    for i in range(max_iter):
        old_edge, new_edge = modify_tree(tree, graph)
        if old_edge is None or new_edge is None:
            continue

        u_old, _ = old_edge
        u_new, _ = new_edge

        delta_weight = (
            tree[new_edge[0]][new_edge[1]]["weight"]
            - tree[old_edge[0]][old_edge[1]]["weight"]
        )
        delta_degree_loss, delta_community_penalty = compute_delta_loss(
            true_out_degrees,
            current_out_degrees,
            u_old,
            u_new,
            num_edges,
            old_edge,
            new_edge,
            communities,
            lambda_c,
        )

        new_loss = (
            current_loss + delta_weight + delta_degree_loss + delta_community_penalty
        )
        delta_loss = new_loss - current_loss

        if delta_loss < 0 or np.exp(-delta_loss / temperature) > random.random():
            current_loss = new_loss
            if new_loss < best_loss:
                best_tree = tree.copy()
                best_loss = new_loss
        else:
            tree.remove_edge(*new_edge)
            tree.add_edge(*old_edge)

        temperature *= cooling_rate
        if temperature < 1e-3:
            break
        if i % 100 == 0:
            print(f"Iteration: {i}, Loss: {best_loss}")

    return best_tree


def alg(graph: nx.Graph, true_tree: nx.Graph):
    communities = perform_leiden(graph)
    nodes_score = nx.betweenness_centrality(graph.to_undirected())
    for u, v in graph.edges():
        graph[u][v]["weight"] = 0.5 * (nodes_score[u] - nodes_score[v]) ** 2
    optimized_tree = simulated_annealing(graph, true_tree, communities)
    return optimized_tree


def load_graph(file_path):
    with open(file_path, "rb") as f:
        graph = pickle.load(f)
    return graph


def save_graph(graph, file_path):
    with open(file_path, "wb") as f:
        pickle.dump(graph, f)
    print(f"Saved tree to {file_path}")


def process_graph(G_path, T_path, S_path):
    graph = load_graph(G_path)
    true_tree = load_graph(T_path)
    result = alg(graph, true_tree)
    save_graph(result, S_path)


def main():
    parser = argparse.ArgumentParser(description="Find optimal directed spanning tree")
    parser.add_argument("G_path", type=str, help="Path to the input graph pkl file")
    parser.add_argument(
        "T_path", type=str, help="Path to the input ground truth tree pkl file"
    )
    parser.add_argument(
        "S_path", type=str, help="Path to save the output tree pkl file"
    )
    args = parser.parse_args()
    process_graph(args.G_path, args.T_path, args.S_path)


if __name__ == "__main__":
    main()
