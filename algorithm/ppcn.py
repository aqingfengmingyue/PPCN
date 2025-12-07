import networkx as nx
import numpy as np
import functools


@functools.lru_cache(maxsize=None)  # No cache limit
def nodes_degree(graph):
    # Return value: dict['Node': 'degree']
    return dict(graph.degree())


@functools.lru_cache(maxsize=None)
def node_to_index(graph: nx.Graph) -> dict[int, int]:
    return {node: i for i, node in enumerate(graph.nodes())}


def valid_degree(G):
    """
    Calculate the valid degree for each node.
    The calculation method is:
    1. Take the node with the maximum degree and its degree value as the initial valid node and its initial valid degree value. The neighbors of the initial node become the "comparison set".
    2. Compare the difference between the neighbors of the remaining nodes and the nodes in the "comparison set". The node with the maximum difference becomes the next valid node, and its valid degree value is the difference value.
    The difference is defined as the number of distinct neighbors a node has compared to the nodes in the "comparison set".
    3. Update the "comparison set" by adding the distinct neighbors of the node identified in step 2.
    4. Repeat steps 2 and 3 to obtain the difference value for each node.
    """

    node_degree: dict[int, int] = nodes_degree(G)

    # Get the node with the maximum degree
    max_node = max(node_degree, key=node_degree.get)

    # Initialize the emerged nodes with the neighbors of the maximum degree node
    emerged_nodes = set(G.neighbors(max_node))
    valid_degree_ = {node: 0 for node in G.nodes()}
    valid_degree_[max_node] = node_degree[max_node]

    transacted_nodes = {max_node}
    untransacted_nodes = set(G.nodes()) - transacted_nodes

    while untransacted_nodes:
        current_valid = {}
        for node in untransacted_nodes:
            # Calculate the current node's validity (difference)
            current_valid[node] = len(set(G.neighbors(node)) - emerged_nodes)

        # Get the node with the maximum validity among unprocessed nodes
        max_node = max(current_valid, key=current_valid.get)
        valid_degree_[max_node] = current_valid[max_node]

        # Update the emerged nodes with the neighbors of the maximum validity node
        emerged_nodes.update(set(G.neighbors(max_node)))

        # Update processed and unprocessed nodes
        transacted_nodes.add(max_node)
        untransacted_nodes.remove(max_node)

        # If all nodes in the network have appeared, set the valid degree of remaining nodes to 0
        if len(emerged_nodes) == len(G.nodes()):
            for node in untransacted_nodes:
                valid_degree_[node] = 0
            return valid_degree_

    return valid_degree_


@functools.lru_cache(maxsize=None)
def valid_degree_with_self_degree(graph: nx.Graph) -> dict[int, int]:
    """
    Calculate the valid degree incorporating the node's own degree.
    Return value: dict{node: value}
    """

    degree = nodes_degree(graph)
    temp_valid_degree = valid_degree(graph)  # Store the valid degree for each node
    node_to_idx = node_to_index(graph)  # dict{node: index}
    improved_valid_degree = {}
    for node in node_to_idx.keys():
        # First improvement of node validity: degree + valid degree
        improved_valid_degree[node] = temp_valid_degree[node] + degree[node]

    return improved_valid_degree


@functools.lru_cache(maxsize=None)
def communicability_matrix_with_belta(graph: nx.Graph, belta: float, distance: int) -> list[list[float]]:
    """
    Calculate the total infection probability from source node to target node considering all paths within a given distance.
    Sum to obtain the final infection probability matrix P, named the communicability matrix, where the self-communicability is 1.
    Pij represents the communicability from node i to node j.
    param: distance determines the propagation path distance, values: 1, 2, 3, 4, 5

    1 - beta**2 represents the probability that node j is not infected by node i via a single distance-2 path.
    **num_path_2 represents the number of all distance-2 paths from node i to node j. Only when
    node j is not infected under all paths, it remains uninfected, so the probability of node j not being infected is
    (probability of not being infected via one path) ** (number of paths). Finally, take the complement to get the probability that node j is infected by node i.
    """
    # Get the shortest paths from node i to all other nodes
    shortest_paths_info = {}
    for node in graph.nodes():
        shortest_paths_info[node] = nx.shortest_path_length(graph, node)

    # Extract neighbors at different layers
    layer_1 = {}
    layer_2 = {}
    layer_3 = {}
    for node in graph.nodes():
        layer_1[node] = set(node for node, distance1 in shortest_paths_info[node].items() if distance1 == 1)
        layer_2[node] = set(node for node, distance2 in shortest_paths_info[node].items() if distance2 == 2)
        layer_3[node] = set(node for node, distance3 in shortest_paths_info[node].items() if distance3 == 3)
    degree = nodes_degree(graph)  # Get the degree of all nodes

    if distance == 1:
        A = nx.to_numpy_array(graph)  # Adjacency matrix of graph
        P = (belta * A)
        np.fill_diagonal(P, 1)

        return P

    elif distance == 2:
        A = nx.to_numpy_array(graph)  # Adjacency matrix of graph
        A_2 = np.linalg.matrix_power(A, 2)
        np.fill_diagonal(A_2, 0)  # Set diagonal elements to 0
        P = (belta * A) + (1 - (1 - belta ** 2) ** A_2)
        np.fill_diagonal(P, 1)

        return P

    elif distance == 3:
        A = nx.to_numpy_array(graph)  # Adjacency matrix of graph
        A_2 = np.linalg.matrix_power(A, 2)
        np.fill_diagonal(A_2, 0)  # Set diagonal elements to 0

        A_3 = A_2 @ A
        node_index = node_to_index(graph)  # dict{node: index}

        # Adjust A_3
        for node, i in node_index.items():
            for neighbor in graph.neighbors(node):
                j = node_index[neighbor]
                A_3[i][j] -= degree[neighbor]
                A_3[i][j] += 1

        np.fill_diagonal(A_3, 0)  # Set A_3 diagonal elements to 0 for later calculations
        P = (belta * A) + (1 - (1 - belta ** 2) ** A_2) + (1 - (1 - belta ** 3) ** A_3)
        np.fill_diagonal(P, 1)

        return P

    elif distance == 4:
        A = nx.to_numpy_array(graph)  # Adjacency matrix of graph
        A_2 = np.linalg.matrix_power(A, 2)
        np.fill_diagonal(A_2, 0)  # Set diagonal elements to 0

        A_3 = A_2 @ A
        node_index = node_to_index(graph)  # dict{node: index}

        # Adjust A_3
        for node, i in node_index.items():
            for neighbor in graph.neighbors(node):
                j = node_index[neighbor]
                A_3[i][j] -= degree[neighbor]
                A_3[i][j] += 1

        np.fill_diagonal(A_3, 0)  # Set A_3 diagonal elements to 0 for later calculations

        A_4 = A_3 @ A
        pretreatment_nodes = {node: layer_1[node] & layer_2[node] for node in graph.nodes()}  # Calculate pretreatment nodes
        for node, i in node_index.items():
            for pending_node in pretreatment_nodes[node]:
                j = node_index[pending_node]
                A_4[i][j] -= degree[pending_node]
                if pending_node in layer_1[node]:
                    A_4[i][j] += 1

        np.fill_diagonal(A_4, 0)
        P = (belta * A) + (1 - (1 - belta ** 2) ** A_2) + (1 - (1 - belta ** 3) ** A_3) + (1 - (1 - belta ** 4) ** A_4)
        np.fill_diagonal(P, 1)
        return P

    elif distance == 5:
        A = nx.to_numpy_array(graph)  # Adjacency matrix of graph
        A_2 = np.linalg.matrix_power(A, 2)
        np.fill_diagonal(A_2, 0)  # Set diagonal elements to 0

        A_3 = A_2 @ A
        node_index = node_to_index(graph)  # dict{node: index}
        # Adjust A_3
        for node, i in node_index.items():
            for neighbor in graph.neighbors(node):
                j = node_index[neighbor]
                A_3[i][j] -= degree[neighbor]
                A_3[i][j] += 1
        np.fill_diagonal(A_3, 0)  # Set A_3 diagonal elements to 0 for later calculations

        A_4 = A_3 @ A
        pretreatment_nodes4 = {node: layer_1[node] & layer_2[node] for node in graph.nodes()}  # Calculate pretreatment nodes
        for node, i in node_index.items():
            for pending_node in pretreatment_nodes4[node]:
                j = node_index[pending_node]
                A_4[i][j] -= degree[pending_node]
                if pending_node in layer_1[node]:
                    A_4[i][j] += 1
        np.fill_diagonal(A_4, 0)  # Set A_4 diagonal elements to 0 for later calculations

        A_5 = A_4 @ A
        pretreatment_nodes5 = {node: layer_1[node] & layer_2[node] & layer_3[node] for node in graph.nodes()}  # Calculate pretreatment nodes
        for node, i in node_index.items():
            for pending_node in pretreatment_nodes5[node]:
                j = node_index[pending_node]
                A_5[i][j] -= degree[pending_node]
                if pending_node in layer_1[node]:
                    A_5[i][j] += 1

        np.fill_diagonal(A_5, 0)  # Set A_5 diagonal elements to 0 for later calculations
        P = (belta * A) + (1 - (1 - belta ** 2) ** A_2) + (1 - (1 - belta ** 3) ** A_3) + (
                    1 - (1 - belta ** 4) ** A_4) + (1 - (1 - belta ** 5) ** A_5)
        np.fill_diagonal(P, 1)

        return P
    else:
        print("Unselected distance")


def contribution_matrix(graph: nx.Graph, belta: float, distance=3) -> float:
    """
    Calculate the node contribution considering the communicability matrix X, and return the contribution matrix.
    Assuming the contribution matrix is C, then Cij represents the contribution of node j to node i.
    """
    selected_distance = distance
    node_index = node_to_index(graph)  # dict{'node': index}
    X = communicability_matrix_with_belta(graph, belta, selected_distance)  # Get communicability matrix X
    improved_valid_degree = valid_degree_with_self_degree(graph)  # dict{node: value}

    # Use broadcasting to directly calculate the contribution matrix
    C = X * np.array([improved_valid_degree[node] for node in node_index.keys()])

    return C


def ppcn(graph: nx.Graph, belta: float, propagated_distance=3):
    """
    Calculate the enhanced valid degree for all nodes, considering the communicability matrix.
    Return value: dict['node', 'value']
    """

    valid_degree_centrality = {}
    node_index = node_to_index(graph)  # dict{'node': index}
    C = contribution_matrix(graph, belta, distance=propagated_distance)  # Get contribution matrix C
    for node, i in node_index.items():
        valid_degree_centrality[node] = round(np.sum(C[i, :]), 2)  # The row sum for index i (corresponding to node) is the final valid degree of node

    return valid_degree_centrality
