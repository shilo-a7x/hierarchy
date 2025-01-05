import argparse
import wikipediaapi
import networkx as nx
import sys
from tqdm import tqdm


def fetch_hierarchy_tree(wiki, category_name, max_depth=sys.maxsize):
    """
    Iteratively fetches the hierarchy tree for a given category.
    Args:
        wiki: Wikipedia API object.
        category_name: Root category name.
        max_depth: Maximum depth of the tree (default is system max int).
    Returns:
        A directed graph representing the hierarchy tree.
    Raises:
        ValueError: If the category does not exist.
    """
    tree = nx.DiGraph()
    stack = [(category_name, 0)]
    visited = set()

    root_category = wiki.page(f"Category:{category_name}")
    if not root_category.exists():
        raise ValueError(
            f"Category '{category_name}' does not exist in the selected Wikipedia language."
        )
    with tqdm(total=1, desc="Processing categories") as pbar:
        while stack:
            current_category, depth = stack.pop()
            if depth > max_depth or current_category in visited:
                continue
            visited.add(current_category)

            category_page = wiki.page(f"Category:{current_category}")
            if not category_page.exists():
                continue

            for member_name, member in category_page.categorymembers.items():
                if member.ns == wikipediaapi.Namespace.CATEGORY:
                    tree.add_edge(
                        current_category, member.title.replace("Category:", "")
                    )
                    stack.append((member.title.replace("Category:", ""), depth + 1))
                elif member.ns == wikipediaapi.Namespace.MAIN:
                    tree.add_edge(current_category, member.title)
            pbar.update(1)

    if tree.number_of_nodes() == 0:
        raise ValueError(
            f"Category '{category_name}' exists but has no valid subcategories or articles."
        )

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
    entity_graph = nx.DiGraph()

    with tqdm(total=tree.number_of_nodes(), desc="Processing articles") as pbar:
        for node in tree.nodes:
            page = wiki.page(node)
            if page.exists():
                for link in page.links.keys():
                    entity_graph.add_edge(node, link)
            pbar.update(1)

    nodes_to_remove = [node for node in entity_graph.nodes if node not in tree.nodes]
    entity_graph.remove_nodes_from(nodes_to_remove)

    entity_graph.remove_edges_from(nx.selfloop_edges(entity_graph))

    if entity_graph.number_of_edges() == 0:
        print("Warning: Entity graph is empty. No links between articles found.")

    return entity_graph


def main():
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
        default=sys.maxsize,
        help="Maximum depth of the hierarchy tree.",
    )
    parser.add_argument(
        "--tree_output",
        type=str,
        default="hierarchy_tree.edgelist",
        help="Output file for the hierarchy tree (edge list).",
    )
    parser.add_argument(
        "--entity_output",
        type=str,
        default="entity_graph.edgelist",
        help="Output file for the entity graph (edge list).",
    )

    args = parser.parse_args()

    try:
        wiki = wikipediaapi.Wikipedia(
            language=args.language,
            user_agent="wikipedia-category-tree-extractor/1.0 (https://github.com/shilo-a7x/hierarchy)",
        )

        print(f"Fetching hierarchy tree for category: {args.category}")
        hierarchy_tree = fetch_hierarchy_tree(wiki, args.category, args.max_depth)
        nx.write_edgelist(hierarchy_tree, args.tree_output, data=False)
        print(f"Hierarchy tree saved to {args.tree_output}")

        print("Building entity graph from hierarchy tree...")
        entity_graph = fetch_entity_graph(hierarchy_tree, wiki)
        nx.write_edgelist(entity_graph, args.entity_output, data=False)
        print(f"Entity graph saved to {args.entity_output}")

    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()
