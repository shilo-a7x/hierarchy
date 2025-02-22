import pickle
import argparse
import random
import networkx as nx
import leidenalg as la
import igraph as ig
import numpy as np
from networkx.algorithms.tree.branchings import Edmonds


def convert_to_igraph(graph: nx.Graph):
    ig_graph = ig.Graph(directed=graph.is_directed())
    ig_graph.add_vertices(len(graph))
    ig_graph.add_edges(list(graph.edges()))
    return ig_graph


def perform_leiden(graph: nx.Graph):
    ig_graph = convert_to_igraph(graph)
    partition = la.find_partition(ig_graph, partition_type=la.ModularityVertexPartition)

    communities = {node: comm for comm, nodes in enumerate(partition) for node in nodes}
    return communities


# def get_random_spanning_tree(G: nx.DiGraph):
#     valid_roots = [node for node in G.nodes() if G.out_degree(node) > 0]
#     if not valid_roots:
#         raise ValueError("Graph has no valid root with outgoing edges")

#     root = random.choice(valid_roots)

#     tree = nx.DiGraph()
#     tree.add_node(root)
#     edges = list(G.edges(data=True))
#     random.shuffle(edges)

#     while len(tree.nodes) < len(G.nodes):
#         for u, v, data in edges:
#             if u in tree.nodes and v not in tree.nodes:
#                 tree.add_edge(u, v, weight=data.get("weight", 1))
#                 if len(tree.nodes) == len(G.nodes):
#                     return tree
#     return tree


# def get_random_arborescence(G: nx.DiGraph, root):
#     if root not in G.nodes:
#         raise ValueError("Given root is not in the graph")
#     if G.out_degree(root) == 0:
#         raise ValueError("Root must have outgoing edges")

#     tree = nx.DiGraph()
#     tree.add_node(root)

#     edges = list(G.edges(data=True))
#     random.shuffle(edges)  # Randomize edge selection

#     visited = set([root])
#     queue = [root]

#     while queue:
#         node = queue.pop(0)
#         candidates = [
#             edge for edge in edges if edge[0] == node and edge[1] not in visited
#         ]
#         random.shuffle(candidates)  # Shuffle candidates to make selection random

#         for u, v, data in candidates:
#             tree.add_edge(u, v, weight=data.get("weight", 1))
#             visited.add(v)
#             queue.append(v)
#             if len(visited) == len(G.nodes):
#                 return tree  # Stop early when all nodes are included

#     if len(visited) < len(G.nodes):
#         raise ValueError("Graph is not fully connected from the chosen root")

#     return tree


def compute_initial_loss(tree: nx.DiGraph, true_tree: nx.DiGraph, communities):
    weight_loss = sum(tree[u][v]["weight"] for u, v in tree.edges())

    true_out_degrees = np.array(list(dict(true_tree.out_degree()).values()))
    current_out_degrees = np.array(list(dict(tree.out_degree()).values()))

    degree_loss = np.mean((true_out_degrees - current_out_degrees) ** 2)

    community_penalty = sum(
        1 for u, v in tree.edges() if communities[u] != communities[v]
    )

    child_diversity = {}
    for parent in tree.nodes():
        child_communities = set(communities[child] for child in tree.successors(parent))
        child_diversity[parent] = len(child_communities)
    diversity_penalty = sum(diversity for diversity in child_diversity.values())

    return (
        weight_loss,
        degree_loss,
        community_penalty,
        diversity_penalty,
        current_out_degrees,
        child_diversity,
    )


def compute_delta_loss(
    tree,
    graph,
    old_edge,
    new_edge,
    true_out_degrees,
    current_out_degrees,
    communities,
    child_diversity,
):
    delta_weight = graph.edges[new_edge]["weight"] - graph.edges[old_edge]["weight"]
    u_old, _ = old_edge
    u_new, _ = new_edge
    old_mse = (
        (true_out_degrees[u_old] - current_out_degrees[u_old]) ** 2
        + (true_out_degrees[u_new] - current_out_degrees[u_new]) ** 2
    ) / tree.number_of_edges()

    current_out_degrees[u_old] -= 1
    current_out_degrees[u_new] += 1

    new_mse = (
        (true_out_degrees[u_old] - current_out_degrees[u_old]) ** 2
        + (true_out_degrees[u_new] - current_out_degrees[u_new]) ** 2
    ) / tree.number_of_edges()

    delta_degree_loss = new_mse - old_mse

    old_penalty = 1 if communities[old_edge[0]] != communities[old_edge[1]] else 0
    new_penalty = 1 if communities[new_edge[0]] != communities[new_edge[1]] else 0
    delta_community_penalty = new_penalty - old_penalty

    new_diversity_old = len(set(communities[child] for child in tree.successors(u_old)))
    new_diversity_new = len(set(communities[child] for child in tree.successors(u_new)))

    delta_diversity_loss = (new_diversity_old - child_diversity[u_old]) + (
        new_diversity_new - child_diversity[u_new]
    )

    return (
        delta_weight,
        delta_degree_loss,
        delta_community_penalty,
        delta_diversity_loss,
        new_diversity_old,
        new_diversity_new,
    )


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
    max_iter=1000000,
    initial_temp=100,
    cooling_rate=0.99,
):
    print("Starting simulated annealing")
    tree = Edmonds(graph).find_optimum(attr="weight", kind="min", style="arborescence")
    print("Initial tree is an arborescence: ", nx.is_arborescence(tree))
    # root = next(node for node in true_tree.nodes() if true_tree.in_degree(node) == 0)
    # tree = get_random_arborescence(graph, root)
    print("Initial tree generated")
    best_tree = tree.copy()

    (
        weight_loss,
        degree_loss,
        community_penalty,
        diversity_penalty,
        current_out_degrees,
        child_diversity,
    ) = compute_initial_loss(tree, true_tree, communities)
    current_loss = weight_loss + degree_loss + community_penalty + diversity_penalty
    best_loss = current_loss

    temperature = initial_temp
    true_out_degrees = dict(true_tree.out_degree())

    effective_modifications = 0
    valid_modifications = 0

    for i in range(max_iter):
        old_edge, new_edge = modify_tree(tree, graph)
        if old_edge is None or new_edge is None:
            continue
        valid_modifications += 1
        (
            delta_weight,
            delta_degree_loss,
            delta_community_penalty,
            delta_diversity_loss,
            new_diversity_old,
            new_diversity_new,
        ) = compute_delta_loss(
            tree,
            graph,
            old_edge,
            new_edge,
            true_out_degrees,
            current_out_degrees,
            communities,
            child_diversity,
        )

        new_loss = (
            current_loss
            + delta_weight
            + delta_degree_loss
            + delta_community_penalty
            + delta_diversity_loss
        )
        delta_loss = new_loss - current_loss

        if delta_loss < 0 or np.exp(-delta_loss / temperature) > random.random():
            current_loss = new_loss
            child_diversity[old_edge[0]] = new_diversity_old
            child_diversity[new_edge[0]] = new_diversity_new
            effective_modifications += 1
            if new_loss < best_loss:
                best_tree = tree.copy()
                best_loss = new_loss
        else:
            tree.remove_edge(*new_edge)
            tree.add_edge(*old_edge)

        temperature *= cooling_rate
        if temperature < 1e-3:
            break
        print(
            f"Iteration: {i}, "
            f"Valid modifications: {valid_modifications}, "
            f"Effective modifications: {effective_modifications}, "
            f"Loss: {best_loss}"
        )
    print(f"Final loss: {best_loss}")
    print(f"Final temperature: {temperature}")
    print(f"Number of iterations: {i}")
    print(f"Valid modifications: {valid_modifications}")
    print(f"Effective modifications: {effective_modifications}")
    print("Simulated annealing completed")
    return best_tree


def alg(graph: nx.Graph, true_tree: nx.Graph):
    communities = perform_leiden(graph)
    nodes_score = nx.betweenness_centrality(graph.to_undirected())
    for u, v in graph.edges():
        graph[u][v]["weight"] = 0.5 * (nodes_score[u] - nodes_score[v]) ** 2
    optimized_tree = simulated_annealing(
        graph, true_tree, communities, max_iter=int(10e6)
    )
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
