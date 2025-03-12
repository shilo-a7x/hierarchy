import pickle
import json
import copy
import random
from collections import deque
import optuna
import networkx as nx
import leidenalg as la
import igraph as ig
import numpy as np
import logging
import traceback
import matplotlib.pyplot as plt
from networkx.algorithms.tree.branchings import Edmonds


# Setup Logging
logging.basicConfig(
    filename="optuna_log.txt", level=logging.INFO, format="%(asctime)s - %(message)s"
)

# Global storage for graph, node scores, and edge weights
GRAPH_DATA = {}


class Pool:
    def __init__(self, elements=None):
        """Fast O(1) sampling, insertion, and removal."""
        self.item_list = []
        self.item_index = {}
        if elements:
            self.add(*elements)

    def sample(self):
        """Returns a random element in O(1)."""
        return random.choice(self.item_list) if self.item_list else None

    def remove(self, *items):
        """Removes one or more items in O(1) using swap-remove."""
        for item in items:
            if item in self.item_index:
                idx = self.item_index[item]
                last_item = self.item_list[-1]
                self.item_list[idx] = last_item
                self.item_index[last_item] = idx
                self.item_list.pop()
                del self.item_index[item]

    def add(self, *items):
        """Adds one or more items in O(1)."""
        for item in items:
            if item not in self.item_index:
                self.item_index[item] = len(self.item_list)
                self.item_list.append(item)


def load_graph(file_path):
    """Loads a graph from a pickle file."""
    with open(file_path, "rb") as f:
        return pickle.load(f)


def save_graph(graph, file_path):
    """Saves a graph to a pickle file."""
    with open(file_path, "wb") as f:
        pickle.dump(graph, f)


def compute_node_scores(graph):
    """Computes node betweenness centrality scores."""
    return nx.betweenness_centrality(graph.to_undirected())


def precompute_edge_weights(graph, node_scores):
    """Assigns edge weights based on precomputed node scores."""
    for u, v in graph.edges():
        graph[u][v]["weight"] = (node_scores[u] - node_scores[v]) ** 2


def compute_depth_vectors(tree: nx.DiGraph):
    """Computes the depth vectors for each node in the tree using NumPy."""
    num_nodes = len(tree.nodes())

    depth_vectors = {node: np.array([1], dtype=int) for node in tree.nodes()}
    child_counts = np.zeros(num_nodes, dtype=int)
    parent_map = {}

    for parent, child in tree.edges():
        parent_map[child] = parent
        child_counts[parent] += 1

    queue = deque([node for node in tree.nodes() if child_counts[node] == 0])

    while queue:
        node = queue.popleft()
        parent = parent_map.get(node)

        if parent is None:
            continue

        while len(depth_vectors[parent]) <= len(depth_vectors[node]):
            depth_vectors[parent] = np.append(depth_vectors[parent], 0)

        depth_vectors[parent][1 : len(depth_vectors[node]) + 1] += depth_vectors[node]

        child_counts[parent] -= 1
        if child_counts[parent] == 0:
            queue.append(parent)

    return depth_vectors, parent_map


def compute_root_depth_vector(tree, root):
    queue = deque([(root, 0)])
    level_counts = {}

    while queue:
        node, depth = queue.popleft()
        level_counts[depth] = level_counts.get(depth, 0) + 1

        for child in tree.successors(node):
            queue.append((child, depth + 1))

    max_depth = max(level_counts.keys())
    counts_array = np.zeros(max_depth + 1, dtype=int)
    for depth, count in level_counts.items():
        counts_array[depth] = count

    return counts_array


def initialize_global_data():
    """Loads graphs and precomputes node scores & edge weights once."""
    global GRAPH_DATA
    G = load_graph("data/wiki/Graph theory/entity_graph.pkl")
    T = load_graph("data/wiki/Graph theory/hierarchy_tree.pkl")
    node_scores = compute_node_scores(G)  # Expensive operation, done once!
    precompute_edge_weights(G, node_scores)  # Store edge weights

    GRAPH_DATA["graph"] = G
    GRAPH_DATA["true_tree"] = T
    initial_tree = Edmonds(G).find_optimum(
        attr="weight", kind="min", style="arborescence"
    )
    GRAPH_DATA["initial_tree"] = initial_tree
    GRAPH_DATA["edge_sampler"] = Pool(GRAPH_DATA["initial_tree"].edges())
    depth_vectors, parent_map = compute_depth_vectors(initial_tree)
    GRAPH_DATA["depth_vectors"] = depth_vectors
    true_root = next(node for node in T.nodes() if T.in_degree(node) == 0)
    GRAPH_DATA["true_root_depth_vector"] = compute_root_depth_vector(T, true_root)
    GRAPH_DATA["parent_map"] = parent_map


def perform_leiden(graph: nx.Graph, resolution):
    """Performs Leiden community detection with a tunable resolution parameter."""
    ig_graph = ig.Graph(directed=graph.is_directed())
    ig_graph.add_vertices(len(graph))
    ig_graph.add_edges(list(graph.edges()))
    partition = la.find_partition(
        ig_graph, la.RBConfigurationVertexPartition, resolution_parameter=resolution
    )
    return {node: comm for comm, nodes in enumerate(partition) for node in nodes}


def compute_statistics(G, T, S):
    """Computes F1-score and related metrics."""
    TP = FP = FN = TN = 0
    positive_edges = set(T.edges)
    total_edges = set(G.edges())
    negative_edges = total_edges - positive_edges
    illegal_edges = set(S.edges()) - total_edges

    for edge in total_edges:
        u, v = edge
        if edge in positive_edges:
            if S.has_edge(u, v):
                TP += 1
            else:
                FN += 1
        elif edge in negative_edges:
            if S.has_edge(u, v):
                FP += 1
            else:
                TN += 1

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = (
        (2 * precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    return TP, FP, FN, TN, f1_score, len(illegal_edges)


# class LossManager:
#     """Manages loss components efficiently with O(1) delta updates."""

#     def __init__(self, tree, true_tree, communities, graph, config):
#         self.weights = config["loss_weights"]
#         self.graph = graph  # Store original graph for shortcut checks
#         self.communities = communities

#         # Compute initial loss values
#         self.losses = {
#             "weight": self.weights["weight"]
#             * sum(tree[u][v]["weight"] for u, v in tree.edges()),
#             "community": self.weights["community"]
#             * sum(
#                 1 for u, v in tree.edges() if self.communities[u] != self.communities[v]
#             ),
#             "diversity": self.weights["diversity"] * self.compute_diversity_loss(tree),
#             "shortcut": 0,
#         }

#     def compute_diversity_loss(self, tree):
#         """Computes the diversity loss for the initial tree."""
#         loss = 0
#         for parent in tree.nodes():
#             child_communities = [
#                 self.communities[child] for child in tree.successors(parent)
#             ]
#             if child_communities:
#                 counts = np.bincount(child_communities)
#                 loss += 1 - (np.max(counts) / np.sum(counts))
#         return loss

#     def total_loss(self):
#         """Returns the weighted loss sum."""
#         return sum(self.losses.values())

#     def compute_delta_loss(self, tree, old_edges, new_edges):
#         """Computes the change in loss when swapping edges."""
#         (x, y), (u, v) = old_edges
#         (x, v), (u, y) = new_edges

#         # Weight Loss Change
#         weight_xv = (
#             self.graph.edges[x, v]["weight"] if (x, v) in self.graph.edges() else 0
#         )
#         weight_xy = (
#             self.graph.edges[x, y]["weight"] if (x, y) in self.graph.edges() else 0
#         )
#         weight_uy = (
#             self.graph.edges[u, y]["weight"] if (u, y) in self.graph.edges() else 0
#         )
#         weight_uv = (
#             self.graph.edges[u, v]["weight"] if (u, v) in self.graph.edges() else 0
#         )

#         delta_weight = self.weights["weight"] * (
#             (weight_xv - weight_xy) + (weight_uy - weight_uv)
#         )

#         # Community Loss Change
#         old_community_penalty = (self.communities[x] != self.communities[y]) + (
#             self.communities[u] != self.communities[v]
#         )
#         new_community_penalty = (self.communities[x] != self.communities[v]) + (
#             self.communities[u] != self.communities[y]
#         )
#         delta_community = self.weights["community"] * (
#             new_community_penalty - old_community_penalty
#         )

#         # Diversity Loss Change
#         def compute_diversity(parent, exclude_child=None, add_child=None):
#             """Computes diversity for a single parent node, simulating the swap."""
#             child_communities = [
#                 self.communities[child]
#                 for child in tree.successors(parent)
#                 if child != exclude_child
#             ]
#             if add_child is not None:
#                 child_communities.append(self.communities[add_child])
#             if not child_communities:
#                 return 0
#             counts = np.bincount(child_communities)
#             return 1 - (np.max(counts) / np.sum(counts))

#         prev_div_x, prev_div_u = compute_diversity(x), compute_diversity(u)
#         new_div_x = compute_diversity(x, exclude_child=y, add_child=v)  # Simulate swap
#         new_div_u = compute_diversity(u, exclude_child=v, add_child=y)  # Simulate swap

#         delta_diversity = self.weights["diversity"] * (
#             (new_div_x + new_div_u) - (prev_div_x + prev_div_u)
#         )

#         prev_shortcut_penalty = ((x, y) not in self.graph.edges()) + (
#             (u, v) not in self.graph.edges()
#         )
#         new_shortcut_penalty = ((x, v) not in self.graph.edges()) + (
#             (u, y) not in self.graph.edges()
#         )
#         delta_shortcut = self.weights["shortcut"] * (
#             new_shortcut_penalty - prev_shortcut_penalty
#         )

#         return delta_weight, delta_community, delta_diversity, delta_shortcut

#     def apply_update(
#         self, delta_weight, delta_community, delta_diversity, delta_shortcut
#     ):
#         """Applies the computed loss changes only if swap is accepted."""
#         self.losses["weight"] += delta_weight
#         self.losses["community"] += delta_community
#         self.losses["diversity"] += delta_diversity
#         self.losses["shortcut"] += delta_shortcut


class TreeStateManager:
    """Manages loss components and state of tree"""

    def __init__(
        self,
        graph,
        tree,
        initial_tree_edges_sampler,
        communities,
        depth_vectors,
        true_root_depth_vector,
        parent_map,
        config,
    ):
        self.weights = config["loss_weights"]
        self.graph = graph
        self.tree = tree
        self.tree_edges = initial_tree_edges_sampler
        self.communities = communities
        self.depth_vectors = depth_vectors
        self.true_root_depth_vector = true_root_depth_vector
        self.parent_map = parent_map
        self.parent_map = {child: parent for parent, child in tree.edges()}

        # Compute initial loss values
        self.losses = {
            "weight": self.weights["weight"]
            * sum(tree[u][v]["weight"] for u, v in tree.edges()),
            "community": self.weights["community"]
            * sum(
                1 for u, v in tree.edges() if self.communities[u] != self.communities[v]
            ),
            "diversity": self.weights["diversity"] * self.compute_diversity_loss(),
            "depth": self.weights["depth"] * self.compute_depth_loss(),
            "shortcut": 0,
        }

    def compute_diversity_loss(self):
        """Computes the diversity loss for the initial tree."""
        loss = 0
        for parent in self.tree.nodes():
            child_communities = [
                self.communities[child] for child in self.tree.successors(parent)
            ]
            if child_communities:
                counts = np.bincount(child_communities)
                loss += 1 - (np.max(counts) / np.sum(counts))
        return loss

    def compute_depth_loss(self):
        root = next(
            node for node in self.tree.nodes() if self.tree.in_degree(node) == 0
        )
        v1 = self.depth_vectors[root]
        v2 = self.true_root_depth_vector

        max_len = max(len(v1), len(v2))
        v1 = np.pad(v1, (0, max_len - len(v1)))
        v2 = np.pad(v2, (0, max_len - len(v2)))

        return np.mean((v1 - v2) ** 2)

    def total_loss(self):
        """Returns the weighted loss sum."""
        return sum(self.losses.values())

    def compute_delta_loss(self, edge_modification):
        """Computes the change in loss when swapping edges."""
        (x, y), (u, v) = edge_modification["old"]
        (x, v), (u, y) = edge_modification["new"]

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
                for child in self.tree.successors(parent)
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

        self.backup = {}

        def update_depth_vectors(node, sign):
            delta_vector = self.depth_vectors[node]
            shift = 1
            while node in self.parent_map:
                parent = self.parent_map[node]
                if parent not in self.backup:
                    self.backup[parent] = self.depth_vectors[parent].copy()
                if len(self.depth_vectors[parent]) < len(delta_vector) + shift:
                    self.depth_vectors[parent] = np.pad(
                        self.depth_vectors[parent],
                        (
                            0,
                            (
                                len(delta_vector)
                                + shift
                                - len(self.depth_vectors[parent])
                            ),
                        ),
                    )
                self.depth_vectors[parent][shift : shift + len(delta_vector)] += (
                    sign * delta_vector
                )
                shift += 1
                node = parent

        update_depth_vectors(y, -1)
        self.parent_map[y] = u
        update_depth_vectors(y, 1)
        update_depth_vectors(v, -1)
        self.parent_map[v] = x
        update_depth_vectors(v, 1)

        self.parent_map[y] = x
        self.parent_map[v] = u

        delta_depth = (
            self.weights["depth"] * self.compute_depth_loss() - self.losses["depth"]
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

        delta_losses = {
            "weight": delta_weight,
            "community": delta_community,
            "diversity": delta_diversity,
            "depth": delta_depth,
            "shortcut": delta_shortcut,
        }

        return delta_losses

    def modify_tree(self):
        old_edge_1 = self.tree_edges.sample()
        old_edge_2 = self.tree_edges.sample()
        x, y = old_edge_1
        u, v = old_edge_2

        if x == u or x == v or y == u or y == v:
            return False, None

        def has_path_backwards(start, target):
            """Returns True if there's a path from `start` to `target` by walking upwards."""
            while start in self.parent_map:
                start = self.parent_map[start]
                if start == target:
                    return True
            return False

        if has_path_backwards(v, x) or has_path_backwards(y, u):
            return False, None
        new_edge_1, new_edge_2 = (x, v), (u, y)
        return True, {"old": (old_edge_1, old_edge_2), "new": (new_edge_1, new_edge_2)}

    def apply_update(self, delta_losses, edge_modification):
        """Applies the computed loss changes and state changes only if swap is accepted."""
        (x, y), (u, v) = edge_modification["old"]
        (x, v), (u, y) = edge_modification["new"]

        self.tree.remove_edges_from([(x, y), (u, v)])
        self.tree.add_edges_from([(x, v), (u, y)])

        self.parent_map[v] = x
        self.parent_map[y] = u
        self.tree_edges.remove((x, y), (u, v))
        self.tree_edges.add((x, v), (u, y))

        self.losses["weight"] += delta_losses["weight"]
        self.losses["community"] += delta_losses["community"]
        self.losses["diversity"] += delta_losses["diversity"]
        self.losses["depth"] += delta_losses["depth"]
        self.losses["shortcut"] += delta_losses["shortcut"]

    def restore_depth_vectors(self):
        """Restores depth vectors from the backup on swap rejection."""
        for node, backup_vector in self.backup.items():
            self.depth_vectors[node] = backup_vector


# def has_path_backwards(start, target, parent_map):
#     """Returns True if there's a path from `start` to `target` by walking upwards."""
#     while start in parent_map:
#         start = parent_map[start]
#         if start == target:
#             return True
#     return False


# def simulated_annealing(
#     graph, true_tree, initial_tree, communities, config, trial=None
# ):
#     """Runs simulated annealing to find an optimal spanning tree."""
#     print("Starting simulated annealing")
#     tree = initial_tree.copy()
#     loss_manager = LossManager(tree, true_tree, communities, graph, config)

#     temperature = config["initial_temp"]
#     cooling_rate = config["cooling_rate"]
#     tree_edges = Pool(tree.edges())
#     parent_map = {child: parent for parent, child in tree.edges()}

#     effective_modifications = 0
#     valid_modifications = 0

#     for i in range(config["max_iter"]):
#         old_edge_1 = tree_edges.sample()
#         old_edge_2 = tree_edges.sample()
#         x, y = old_edge_1
#         u, v = old_edge_2

#         if x == u or x == v or y == u or y == v:
#             continue

#         if has_path_backwards(v, x, parent_map) or has_path_backwards(y, u, parent_map):
#             continue
#         valid_modifications += 1
#         new_edge_1, new_edge_2 = (x, v), (u, y)

#         delta_weight, delta_community, delta_diversity, delta_shortcut = (
#             loss_manager.compute_delta_loss(
#                 tree, [old_edge_1, old_edge_2], [new_edge_1, new_edge_2]
#             )
#         )
#         new_loss = (
#             loss_manager.total_loss()
#             + delta_weight
#             + delta_community
#             + delta_diversity
#             + delta_shortcut
#         )
#         delta_loss = new_loss - loss_manager.total_loss()

#         if delta_loss < 0 or np.exp(-delta_loss / temperature) > random.random():
#             tree.remove_edges_from([old_edge_1, old_edge_2])
#             tree.add_edges_from([new_edge_1, new_edge_2])
#             parent_map[v] = x
#             parent_map[y] = u
#             tree_edges.remove(old_edge_1)
#             tree_edges.remove(old_edge_2)
#             tree_edges.add(new_edge_1)
#             tree_edges.add(new_edge_2)
#             loss_manager.apply_update(
#                 delta_weight, delta_community, delta_diversity, delta_shortcut
#             )
#             effective_modifications += 1

#         if trial and i % 10000 == 0:
#             current_loss = loss_manager.total_loss()
#             trial.report(current_loss, step=i)

#             # If Optuna suggests pruning, exit early
#             if trial.should_prune():
#                 raise optuna.exceptions.TrialPruned()

#         temperature *= cooling_rate
#         if temperature < 1e-3:
#             break
#     print(
#         "\n".join(
#             [
#                 f"Trial {trial.number} completed",
#                 f"Final loss: {loss_manager.total_loss()}",
#                 f"Final temperature: {temperature}",
#                 f"Number of iterations: {i}",
#                 f"Valid modifications: {valid_modifications}",
#                 f"Effective modifications: {effective_modifications}",
#                 "Simulated annealing completed",
#             ]
#         )
#     )
#     return tree


def simulated_annealing(
    graph,
    initial_tree,
    initial_tree_edges_sampler,
    communities,
    depth_vectors,
    true_root_depth_vector,
    parent_map,
    config,
    trial=None,
):
    """Runs simulated annealing to find an optimal spanning tree."""
    print("Starting simulated annealing...")

    temperature = config["initial_temp"]
    cooling_rate = config["cooling_rate"]

    tree_state_manager = TreeStateManager(
        graph,
        initial_tree,
        initial_tree_edges_sampler,
        communities,
        depth_vectors,
        true_root_depth_vector,
        parent_map,
        config,
    )

    for i in range(config["max_iter"]):
        valid, edge_modification = tree_state_manager.modify_tree()
        if not valid:
            continue

        delta_losses = tree_state_manager.compute_delta_loss(edge_modification)
        delta_loss = sum(delta_losses.values())

        if delta_loss < 0 or np.exp(-delta_loss / temperature) > random.random():
            tree_state_manager.apply_update(delta_losses, edge_modification)
        else:
            tree_state_manager.restore_depth_vectors()

        if trial and i % 10000 == 0:
            current_loss = tree_state_manager.total_loss()
            trial.report(current_loss, step=i)

            # If Optuna suggests pruning, exit early
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        temperature *= cooling_rate
        if temperature < 1e-3:
            break
    print(
        "\n".join(
            [
                f"Trial {trial.number} completed",
                f"Final loss: {tree_state_manager.total_loss()}",
                f"Final temperature: {temperature}",
                f"Number of iterations: {i}",
                "Simulated annealing completed",
            ]
        )
    )
    return tree_state_manager.tree


def alg(graph, true_tree, config, trial):
    communities = perform_leiden(graph, config["resolution"])
    return simulated_annealing(graph, true_tree, communities, config, trial)


def objective(trial):
    """Objective function for Optuna optimization."""
    config = {
        "initial_temp": trial.suggest_float("initial_temp", 100, 1000),
        "cooling_rate": trial.suggest_float("cooling_rate", 0.999, 0.99999),
        "max_iter": trial.suggest_int("max_iter", int(1e5), int(1e9), step=int(1e5)),
        "loss_weights": {
            "weight": trial.suggest_float("weight", 1, 100),
            "community": trial.suggest_float("community", 1, 100),
            "diversity": trial.suggest_float("diversity", 1, 100),
            "depth": trial.suggest_float("depth", 1, 100),
            "shortcut": trial.suggest_float("shortcut", 1, 1e4),
        },
        "resolution": trial.suggest_float("resolution", 0.1, 5),
    }

    try:
        G = GRAPH_DATA["graph"]
        T = GRAPH_DATA["true_tree"]
        S_0 = copy.deepcopy(GRAPH_DATA["initial_tree"])
        edge_sampler = copy.deepcopy(GRAPH_DATA["edge_sampler"])
        communities = perform_leiden(G, config["resolution"])
        depth_vectors = copy.deepcopy(GRAPH_DATA["depth_vectors"])
        true_root_depth_vector = GRAPH_DATA["true_root_depth_vector"]
        parent_map = copy.deepcopy(GRAPH_DATA["parent_map"])
        S = simulated_annealing(
            G,
            S_0,
            edge_sampler,
            communities,
            depth_vectors,
            true_root_depth_vector,
            parent_map,
            config,
            trial,
        )  # Run simulated annealing with given parameters

        TP, FP, FN, TN, f1_score, shortcuts = compute_statistics(G, T, S)
        logging.info(
            f"Trial {trial.number}: F1-score = {f1_score}, Shortcuts = {shortcuts}, TP = {TP}, Parameters = {config}"
        )

        return TP

    except optuna.exceptions.TrialPruned:
        logging.info(f"Trial {trial.number} pruned due to early stopping.")
        raise

    except Exception as e:
        logging.error(f"Trial {trial.number} failed with error: {e}")
        traceback.print_exc()
        return 0  # Return worst score so Optuna discards it


if __name__ == "__main__":
    initialize_global_data()
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100, n_jobs=-1)  # Use all CPUs

    best_params = study.best_params
    best_TP = study.best_value

    best_config = {
        "initial_temp": best_params["initial_temp"],
        "cooling_rate": best_params["cooling_rate"],
        "max_iter": best_params["max_iter"],
        "loss_weights": {
            "weight": best_params["weight"],
            "community": best_params["community"],
            "diversity": best_params["diversity"],
            "depth": best_params["depth"],
            "shortcut": best_params["shortcut"],
        },
        "resolution": best_params["resolution"],
    }

    with open("best_hyperparameters.json", "w") as f:
        json.dump(best_config, f, indent=4)

    logging.info(f"Best parameters: {best_params}, Best TP: {best_TP}")

    # Generate & save plots
    optuna.visualization.matplotlib.plot_param_importances(study)
    plt.savefig("param_importance.png")

    optuna.visualization.matplotlib.plot_optimization_history(study)
    plt.savefig("optimization_history.png")

    optuna.visualization.matplotlib.plot_parallel_coordinate(study)
    plt.savefig("parallel_coordinates.png")

    print("Optimization finished. Results saved.")
