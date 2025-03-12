import pickle
import json
import argparse
import random
import networkx as nx
import leidenalg as la
import igraph as ig
import numpy as np
from networkx.algorithms.tree.branchings import Edmonds


class Pool:
    def __init__(self, elements=None):
        """Fast O(1) sampling, insertion, and removal."""
        self.item_list = []
        self.item_index = {}
        if elements:
            for item in elements:
                self.add(item)

    def sample(self):
        """Returns a random element in O(1)."""
        return random.choice(self.item_list) if self.item_list else None

    def remove(self, item):
        """Removes an item in O(1) using swap-remove."""
        if item not in self.item_index:
            return
        idx = self.item_index[item]
        last_item = self.item_list[-1]
        self.item_list[idx] = last_item
        self.item_index[last_item] = idx
        self.item_list.pop()
        del self.item_index[item]

    def add(self, item):
        """Adds an item in O(1)."""
        if item in self.item_index:
            return
        self.item_index[item] = len(self.item_list)
        self.item_list.append(item)


def convert_to_igraph(graph: nx.Graph):
    ig_graph = ig.Graph(directed=graph.is_directed())
    ig_graph.add_vertices(len(graph))
    ig_graph.add_edges(list(graph.edges()))
    return ig_graph


def perform_leiden(graph: nx.Graph):
    ig_graph = convert_to_igraph(graph)
    partition = la.find_partition(ig_graph, partition_type=la.ModularityVertexPartition)
    return {node: comm for comm, nodes in enumerate(partition) for node in nodes}


class LossManager:
    """Manages loss components efficiently with O(1) delta updates."""

    def __init__(self, tree, true_tree, communities, graph, config):
        self.weights = config["loss_weights"]
        self.graph = graph  # Store original graph for shortcut checks
        self.communities = communities

        # Compute initial loss values
        self.losses = {
            "weight": self.weights["weight"]
            * sum(tree[u][v]["weight"] for u, v in tree.edges()),
            "community": self.weights["community"]
            * sum(
                1 for u, v in tree.edges() if self.communities[u] != self.communities[v]
            ),
            "diversity": self.weights["diversity"] * self.compute_diversity_loss(tree),
            "shortcut": 0,
        }

    def compute_diversity_loss(self, tree):
        """Computes the diversity loss for the initial tree."""
        loss = 0
        for parent in tree.nodes():
            child_communities = [
                self.communities[child] for child in tree.successors(parent)
            ]
            if child_communities:
                counts = np.bincount(child_communities)
                loss += 1 - (np.max(counts) / np.sum(counts))
        return loss

    def total_loss(self):
        """Returns the weighted loss sum."""
        return sum(self.losses.values())

    def compute_delta_loss(self, tree, old_edges, new_edges):
        """Computes the change in loss when swapping edges."""
        (x, y), (u, v) = old_edges
        (x, v), (u, y) = new_edges

        # Weight Loss Change
        weight_xv = (
            self.graph.edges[x, v]["weight"] if (x, v) in self.graph.edges() else 0
        )
        weight_xy = (
            self.graph.edges[x, y]["weight"] if (x, y) in self.graph.edges() else 0
        )
        weight_uy = (
            self.graph.edges[u, y]["weight"] if (u, y) in self.graph.edges() else 0
        )
        weight_uv = (
            self.graph.edges[u, v]["weight"] if (u, v) in self.graph.edges() else 0
        )

        delta_weight = self.weights["weight"] * (
            (weight_xv - weight_xy) + (weight_uy - weight_uv)
        )

        # Community Loss Change
        old_community_penalty = (self.communities[x] != self.communities[y]) + (
            self.communities[u] != self.communities[v]
        )
        new_community_penalty = (self.communities[x] != self.communities[v]) + (
            self.communities[u] != self.communities[y]
        )
        delta_community = self.weights["community"] * (
            new_community_penalty - old_community_penalty
        )

        # Diversity Loss Change
        def compute_diversity(parent, exclude_child=None, add_child=None):
            """Computes diversity for a single parent node, simulating the swap."""
            child_communities = [
                self.communities[child]
                for child in tree.successors(parent)
                if child != exclude_child
            ]
            if add_child is not None:
                child_communities.append(self.communities[add_child])
            if not child_communities:
                return 0
            counts = np.bincount(child_communities)
            return 1 - (np.max(counts) / np.sum(counts))

        prev_div_x, prev_div_u = compute_diversity(x), compute_diversity(u)
        new_div_x = compute_diversity(x, exclude_child=y, add_child=v)  # Simulate swap
        new_div_u = compute_diversity(u, exclude_child=v, add_child=y)  # Simulate swap

        delta_diversity = self.weights["diversity"] * (
            (new_div_x + new_div_u) - (prev_div_x + prev_div_u)
        )

        prev_shortcut_penalty = ((x, y) not in self.graph.edges()) + (
            (u, v) not in self.graph.edges()
        )
        new_shortcut_penalty = ((x, v) not in self.graph.edges()) + (
            (u, y) not in self.graph.edges()
        )
        delta_shortcut = self.weights["shortcut"] * (
            new_shortcut_penalty - prev_shortcut_penalty
        )

        return delta_weight, delta_community, delta_diversity, delta_shortcut

    def apply_update(
        self, delta_weight, delta_community, delta_diversity, delta_shortcut
    ):
        """Applies the computed loss changes only if swap is accepted."""
        self.losses["weight"] += delta_weight
        self.losses["community"] += delta_community
        self.losses["diversity"] += delta_diversity
        self.losses["shortcut"] += delta_shortcut


def has_path_backwards(start, target, parent_map):
    """Returns True if there's a path from `start` to `target` by walking upwards."""
    while start in parent_map:
        start = parent_map[start]
        if start == target:
            return True
    return False


def simulated_annealing(graph, true_tree, communities, config):
    """Runs simulated annealing to find an optimal spanning tree."""
    print("Starting simulated annealing")
    tree = Edmonds(graph).find_optimum(attr="weight", kind="min", style="arborescence")
    loss_manager = LossManager(tree, true_tree, communities, graph, config)

    temperature = config["initial_temp"]
    cooling_rate = config["cooling_rate"]
    tree_edges = Pool(tree.edges())
    parent_map = {child: parent for parent, child in tree.edges()}

    effective_modifications = 0
    valid_modifications = 0

    for i in range(config["max_iter"]):
        old_edge_1 = tree_edges.sample()
        old_edge_2 = tree_edges.sample()
        x, y = old_edge_1
        u, v = old_edge_2

        if x == u or x == v or y == u or y == v:
            continue

        if has_path_backwards(v, x, parent_map) or has_path_backwards(y, u, parent_map):
            continue
        valid_modifications += 1
        new_edge_1, new_edge_2 = (x, v), (u, y)

        delta_weight, delta_community, delta_diversity, delta_shortcut = (
            loss_manager.compute_delta_loss(
                tree, [old_edge_1, old_edge_2], [new_edge_1, new_edge_2]
            )
        )
        new_loss = (
            loss_manager.total_loss()
            + delta_weight
            + delta_community
            + delta_diversity
            + delta_shortcut
        )
        delta_loss = new_loss - loss_manager.total_loss()

        if delta_loss < 0 or np.exp(-delta_loss / temperature) > random.random():
            tree.remove_edges_from([old_edge_1, old_edge_2])
            tree.add_edges_from([new_edge_1, new_edge_2])
            parent_map[v] = x
            parent_map[y] = u
            tree_edges.remove(old_edge_1)
            tree_edges.remove(old_edge_2)
            tree_edges.add(new_edge_1)
            tree_edges.add(new_edge_2)
            loss_manager.apply_update(
                delta_weight, delta_community, delta_diversity, delta_shortcut
            )
            effective_modifications += 1

        temperature *= cooling_rate
        if i % 1000 == 0:
            print(
                f"Iteration: {i}, "
                f"Valid modifications: {valid_modifications}, "
                f"Effective modifications: {effective_modifications}, "
                f"Loss: {loss_manager.total_loss()}"
            )
        if temperature < 1e-3:
            break
    print(f"Final loss: {loss_manager.total_loss()}")
    print(f"Final temperature: {temperature}")
    print(f"Number of iterations: {i}")
    print(f"Valid modifications: {valid_modifications}")
    print(f"Effective modifications: {effective_modifications}")
    print("Simulated annealing completed")
    return tree


def alg(graph, true_tree, config):
    communities = perform_leiden(graph)
    nodes_score = nx.betweenness_centrality(graph.to_undirected())
    for u, v in graph.edges():
        graph[u][v]["weight"] = (nodes_score[u] - nodes_score[v]) ** 2
    return simulated_annealing(graph, true_tree, communities, config)


def load_graph(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f)


def save_graph(graph, file_path):
    with open(file_path, "wb") as f:
        pickle.dump(graph, f)


def load_config(config_path):
    with open(config_path, "r") as f:
        return json.load(f)


def process_graph(G_path, T_path, S_path, config):
    graph = load_graph(G_path)
    true_tree = load_graph(T_path)
    result = alg(graph, true_tree, config)
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
    parser.add_argument(
        "--config", type=str, default=None, help="Path to the config file"
    )
    args = parser.parse_args()
    if args.config:
        config = load_config(args.config)
    else:
        config = {
            "max_iter": int(1e6),
            "initial_temp": 1000,
            "cooling_rate": 0.99999,
            "loss_weights": {
                "weight": 1,
                "community": 1,
                "diversity": 1,
                "shortcut": 10,
            },
        }
    process_graph(args.G_path, args.T_path, args.S_path, config)


if __name__ == "__main__":
    main()
