import os
import pickle
import networkx as nx
import random


def triangular_int_sample(min_val, max_val):
    mode = (min_val + max_val) / 2
    sample = random.triangular(min_val, max_val, mode)
    return max(min_val, min(max_val, round(sample)))


def generate_tree(num_vertices, max_children=5):
    tree = nx.DiGraph()
    tree.add_node(0)
    current_nodes = [0]
    next_node_id = 1
    while next_node_id < num_vertices:
        parent = random.choice(current_nodes)
        children = triangular_int_sample(1, max_children)
        for _ in range(children):
            if next_node_id < num_vertices:
                tree.add_edge(parent, next_node_id, type="hierachy")
                current_nodes.append(next_node_id)
                next_node_id += 1
        current_nodes.remove(parent)

    return tree


def add_random_edges(graph, percentage):
    num_existing_edges = graph.number_of_edges()
    num_edges_to_add = int((percentage / 100) * num_existing_edges)

    num_nodes = len(graph.nodes)

    current_edges = set(graph.edges)
    added_edges = 0

    while added_edges < num_edges_to_add:
        u, v = random.sample(range(num_nodes), 2)
        if (u, v) in current_edges or (v, u) in current_edges:
            continue
        graph.add_edge(u, v, type="non-hierarchy")
        current_edges.add((u, v))
        added_edges += 1

    return graph


def save_graph(graph, folder, file_name):
    os.makedirs(folder, exist_ok=True)
    file_path = os.path.join(folder, file_name)
    with open(file_path, "wb") as f:
        pickle.dump(graph, f)
    print(f"Saved graph to {file_path}")


num_vertices_list = [100, 500, 1000]

percentages = [50, 100, 600, 1100]

for num_vertices in num_vertices_list:
    original_tree = generate_tree(num_vertices)

    data_folder = os.path.join("data", "synthetic", f"{num_vertices}")
    os.makedirs(data_folder, exist_ok=True)

    for percentage in percentages:
        graph_with_edges = add_random_edges(original_tree.copy(), percentage)
        file_name = f"G_{num_vertices}_{percentage}.pkl"
        save_graph(graph_with_edges, data_folder, file_name)
