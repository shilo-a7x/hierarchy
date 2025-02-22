import argparse
import wikipediaapi
import networkx as nx
import os
import sys
import pickle


def save_graph(graph, output_path, name):
    """
    Save the graph in multiple formats: edgelist, JSON, and pickle.
    Args:
        graph: NetworkX graph to save.
        output_path: Base directory for saving files.
        name: Name prefix for the output files.
    """
    edgelist_path = os.path.join(output_path, f"{name}.edgelist")
    nx.write_edgelist(graph, edgelist_path, data=False, delimiter="|")
    print(f"Saved {name} as edgelist: {edgelist_path}")

    pickle_path = os.path.join(output_path, f"{name}.pkl")
    with open(pickle_path, "wb") as f:
        pickle.dump(graph, f)
    print(f"Saved {name} as Pickle: {pickle_path}")


def fetch_hierarchy_tree(wiki, category_name, max_depth=5):
    """
    Fetches the Wikipedia hierarchy tree, ensuring it is an arborescence.
    Args:
        wiki: Wikipedia API object.
        category_name: Root category name.
        max_depth: Maximum depth of the tree.
    Returns:
        A directed graph representing the hierarchy tree.
    Raises:
        ValueError: If the category does not exist.
    """
    tree = nx.DiGraph()
    stack = [(category_name, 0)]
    visited = set()
    parent_map = {}
    num_categories = 1

    root_category = wiki.page(f"Category:{category_name}")
    if not root_category.exists():
        raise ValueError(
            f"Category '{category_name}' does not exist in the selected Wikipedia language."
        )

    while stack:
        current_category, depth = stack.pop()
        if depth > max_depth or current_category in visited:
            continue
        visited.add(current_category)

        category_page = wiki.page(f"Category:{current_category}")
        if not category_page.exists():
            continue

        for member_name, member in category_page.categorymembers.items():
            child_name = member.title.replace("Category:", "")

            if member.ns == wikipediaapi.Namespace.CATEGORY:
                if child_name != current_category and child_name not in parent_map:
                    tree.add_edge(current_category, child_name)
                    parent_map[child_name] = current_category
                    stack.append((child_name, depth + 1))
                    num_categories += 1
            elif member.ns == wikipediaapi.Namespace.MAIN:
                if member.title != current_category and member.title not in parent_map:
                    tree.add_edge(current_category, member.title)
                    parent_map[member.title] = current_category

    if tree.number_of_nodes() == 0:
        raise ValueError(
            f"Category '{category_name}' exists but has no valid subcategories or articles."
        )

    print(f"Number of categories in the hierarchy tree: {num_categories}")
    print(f"Number of nodes in the hierarchy tree: {tree.number_of_nodes()}")
    return tree


def fetch_entity_graph(tree, wiki):
    """
    Builds the entity graph from the hierarchy tree.
    Args:
        tree: The hierarchy tree as a directed graph.
        wiki: Wikipedia API object.
    Returns:
        A directed graph representing the entity graph.
    """
    entity_graph = tree.copy()

    for node in entity_graph.nodes:
        page = wiki.page(node)
        if page.exists():
            for link in page.links.keys():
                if link in tree.nodes:
                    entity_graph.add_edge(node, link)

    entity_graph.remove_edges_from(nx.selfloop_edges(entity_graph))

    if entity_graph.number_of_edges() == 0:
        print("Warning: Entity graph is empty. No links between articles found.")

    print(f"Number of edges in the entity graph: {entity_graph.number_of_edges()}")

    return entity_graph


def main():
    # default_output_path = os.path.dirname(os.path.abspath(__file__))
    default_output_path = "data/wiki"

    parser = argparse.ArgumentParser(
        description="Generate hierarchy tree and entity graph from Wikipedia categories."
    )
    parser.add_argument(
        "--language",
        type=str,
        default="en",
        help="Language edition of Wikipedia (default: 'en').",
    )
    parser.add_argument(
        "--category",
        type=str,
        required=True,
        help="Root category to fetch the hierarchy tree.",
    )
    parser.add_argument(
        "--max_depth",
        type=int,
        default=5,
        help="Maximum depth of the hierarchy tree (use --use_max_depth for unlimited depth).",
    )
    parser.add_argument(
        "--use_max_depth",
        action="store_true",
        help="If set, overrides --max_depth to use the system's maximum integer size.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=default_output_path,
        help=f"Base path to save the processed data (default: '{default_output_path}').",
    )

    args = parser.parse_args()
    max_depth = sys.maxsize if args.use_max_depth else args.max_depth

    try:
        wiki = wikipediaapi.Wikipedia(
            language=args.language,
            user_agent="wikipedia-category-tree-extractor/1.0 (https://github.com/shilo-a7x/hierarchy)",
        )
        category_dir = os.path.join(args.output_path, args.category)
        os.makedirs(category_dir, exist_ok=True)

        print(f"Fetching hierarchy tree for category: {args.category}")
        hierarchy_tree = fetch_hierarchy_tree(wiki, args.category, max_depth)
        # tree_name = (
        #     f"hierarchy_tree_depth_{args.max_depth}"
        #     if args.max_depth < sys.maxsize
        #     else "hierarchy_tree"
        # )
        tree_name = "hierarchy_tree"
        save_graph(hierarchy_tree, category_dir, tree_name)
        print("Building entity graph from hierarchy tree...")
        entity_graph = fetch_entity_graph(hierarchy_tree, wiki)
        save_graph(entity_graph, category_dir, "entity_graph")

    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()
