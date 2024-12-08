import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import cm
import math
import numpy as np
import community as community_louvain  # Louvain method for community detection

# Step 1: Load the graph data from the .mtx file
filename = "data/tech-routers-rf.mtx"  # Replace with the path to your .mtx file

# Read edges from the .mtx file and build an undirected graph
edges = []
with open(filename, "r") as file:
    for line in file:
        # Skip comments
        if line.startswith("%"):
            continue
        # Parse edge data only if the line has exactly two values
        values = line.strip().split()
        if len(values) == 2:
            i, j = map(int, values)
            edges.append((i - 1, j - 1))  # Convert to 0-based indexing for NetworkX

# Create the graph
G = nx.Graph()
G.add_edges_from(edges)

# Step 2: Identify the largest connected component
largest_component = max(nx.connected_components(G), key=len)
largest_subgraph = G.subgraph(largest_component).copy()

# Step 3: Calculate Betweenness Centrality for each node in the largest component
betweenness_centrality = nx.betweenness_centrality(largest_subgraph)
min_score = min(betweenness_centrality.values())
max_score = max(betweenness_centrality.values())
centrality_scores = list(betweenness_centrality.values())

# Choose a base for exponential binning (e.g., r = 1.5)
r = 10

# Create exponentially spaced bins
bin_edges = [max_score]
while bin_edges[-1] > min_score:
    bin_edges.append(bin_edges[-1] / r)

# Assign hierarchy levels based on bins
hierarchy = {}
for node, centrality in betweenness_centrality.items():
    for i, bin_edge in enumerate(bin_edges[:-1]):
        if bin_edges[i] <= centrality < bin_edges[i + 1]:
            hierarchy[node] = i
            break


# # Step 5: Define a Threshold Function for Binning
# def assign_hierarchy_bins(centrality_scores, threshold=0.05):
#     sorted_nodes = sorted(centrality_scores.items(), key=lambda x: -x[1])
#     hierarchy = {}
#     current_level = 0
#     last_score = None

#     for node, score in sorted_nodes:
#         if last_score is None or abs(score - last_score) > threshold:
#             current_level += 1
#         hierarchy[node] = current_level
#         last_score = score

#     return hierarchy


# # Assign hierarchy levels based on normalized centrality with thresholding
# hierarchy = assign_hierarchy_bins(betweenness_centrality, threshold=threshold)

# Step 6: Create a DAG from the hierarchy levels
DAG = nx.DiGraph()
for u, v in largest_subgraph.edges():
    if hierarchy[u] < hierarchy[v]:
        DAG.add_edge(u, v)
    elif hierarchy[u] > hierarchy[v]:
        DAG.add_edge(v, u)

# Output the resulting hierarchy and visualize the DAG (optional)
print(
    "Hierarchy Levels by Betweenness Centrality with Thresholding:",
    hierarchy,
)

# Optional: Check if the resulting graph is a DAG
is_dag = nx.is_directed_acyclic_graph(DAG)
print("Is the resulting graph a DAG?", is_dag)

# # Step 6: Visualize the graph with different colors for each hierarchy level
# # Map hierarchy levels to colors
# node_colors = [hierarchy[node] for node in largest_subgraph.nodes()]

# # Define the layout for node positioning
# pos = nx.spring_layout(largest_subgraph, seed=42)

# # Plot the graph
# plt.figure(figsize=(12, 12))
# nx.draw(
#     largest_subgraph,
#     pos,
#     with_labels=True,
#     node_size=100,
#     node_color=node_colors,
#     cmap=plt.colormaps["viridis"],
#     font_size=8,
#     edge_color="gray",
# )

# # Show the plot
# plt.title("Graph with Hierarchy Levels (Colored)")
# plt.show()


# # Step 7: Community Detection using Louvain Method
# # Note: Convert DAG back to undirected for community detection, since Louvain method requires undirected graph
# undirected_DAG = DAG.to_undirected()

# # Perform community detection using Louvain method
# partition = community_louvain.best_partition(undirected_DAG)

# # Step 8: Remove edges between communities in the DAG
# # Create a new directed graph without inter-community edges
# filtered_DAG = nx.DiGraph()

# for u, v in DAG.edges():
#     if partition[u] == partition[v]:
#         filtered_DAG.add_edge(u, v)

# # Output the number of communities detected and the edges removed
# num_communities = len(set(partition.values()))
# print(f"Number of communities detected: {num_communities}")
# print(f"Number of edges before filtering: {len(DAG.edges())}")
# print(f"Number of edges after filtering: {len(filtered_DAG.edges())}")

# Optional: Visualize the filtered DAG
# import matplotlib.pyplot as plt
# pos = nx.spring_layout(filtered_DAG)
# nx.draw(filtered_DAG, pos, with_labels=True, node_size=100, font_size=8, edge_color="blue")
# plt.show()
