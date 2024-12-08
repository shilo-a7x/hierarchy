import os
import pickle
import networkx as nx
from networkx.algorithms.community import louvain_communities
import argparse


def graph_nodes_score(graph, score=None):
    centrality = nx.betweenness_centrality(graph.to_undirected())
    return centrality


def dag_from_score(graph, score) -> nx.DiGraph:
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


def alg(graph: nx.Graph):
    graph_score = graph_nodes_score(graph)
    dag = dag_from_score(graph, graph_score)
    communities = louvain_communities(dag.to_undirected(), threshold=0.8)
    community_subgraphs = [dag.subgraph(community).copy() for community in communities]
    hierarchy_tree = tree_from_subdags(community_subgraphs, graph_score)
    print(nx.is_tree(hierarchy_tree))
    return hierarchy_tree


def load_graph(file_path):
    with open(file_path, "rb") as f:
        graph = pickle.load(f)
    return graph


def save_graph(graph, file_path):
    with open(file_path, "wb") as f:
        pickle.dump(graph, f)
    print(f"Saved tree to {file_path}")


def process_graph(input_file, output_file):
    graph = load_graph(input_file)
    result_tree = alg(graph)
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

    args = parser.parse_args()

    process_graph(args.input_file, args.output_file)


if __name__ == "__main__":
    main()
