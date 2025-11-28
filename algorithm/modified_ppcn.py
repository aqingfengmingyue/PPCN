import networkx as nx
import numpy as np
import functools


@functools.lru_cache(maxsize=None)  # 没有缓存限制
def nodes_degree(graph):
    #  返回值：dict['Node': 'degree']
    return dict(graph.degree())


@functools.lru_cache(maxsize=None)
def node_to_index(graph: nx.Graph) -> dict[int, int]:
    return {node: i for i, node in enumerate(graph.nodes())}


def valid_degree(G):
    """
    计算节点有效度。
    计算方式为：
    1.将最大度值及其对应节点作为有效度的初始值及其初始节点。并将初始节点的邻居节点作为“用于比较的对象”。
    2.比较剩余节点的邻居节点与“用于比较的对象”中节点的差异性，将具有最大差异性的节点作为有效节点中的下一个节点，其有效度的值为差异性的值。
    差异性是指自身邻居节点中与用于较的对象中不同节点的个数。
    3.将2中所确定节点的邻居节点中的差异性节点更新到“用于比较的对象”。
    4.重复2和3获取每个节点的差异性值
    """

    node_degree: dict[int, int] = nodes_degree(G)

    # 获取具有最大度值的节点
    max_node = max(node_degree, key=node_degree.get)

    # 将度数最大的节点作为初始化出现节点
    emerged_nodes = set(G.neighbors(max_node))
    valid_degree_ = {node: 0 for node in G.nodes()}
    valid_degree_[max_node] = node_degree[max_node]

    transacted_nodes = {max_node}
    untransacted_nodes = set(G.nodes()) - transacted_nodes

    while untransacted_nodes:
        current_valid = {}
        for node in untransacted_nodes:
            # 计算当前节点的有效性
            current_valid[node] = len(set(G.neighbors(node)) - emerged_nodes)

        # 获取未处理节点中具有最大有效值的节点
        max_node = max(current_valid, key=current_valid.get)
        valid_degree_[max_node] = current_valid[max_node]

        # 将最大有效性节点的邻居节点更新到已出现的节点里
        emerged_nodes.update(set(G.neighbors(max_node)))

        # 更新已经处理过的和未处理过的节点
        transacted_nodes.add(max_node)
        untransacted_nodes.remove(max_node)

        # 当网络中所有节点都已经出现过则令之后节点的有效度值全为0
        if len(emerged_nodes) == len(G.nodes()):
            for node in untransacted_nodes:
                valid_degree_[node] = 0
            return valid_degree_

    return valid_degree_


@functools.lru_cache(maxsize=None)
def valid_degree_with_self_degree(graph: nx.Graph) -> dict[int, int]:
    """
    计算带有自身度值的有效度
    返回值：dict{node:value}
    """

    degree = nodes_degree(graph)
    temp_valid_degree = valid_degree(graph)  # 存储每个节点的有效度
    node_to_idx = node_to_index(graph)  # dict{'node':index}
    improved_valid_degree = {}
    for node in node_to_idx.keys():
        # 第一次改进节点有效性，度值+有效值
        improved_valid_degree[node] = temp_valid_degree[node] + degree[node]

    return improved_valid_degree


@functools.lru_cache(maxsize=None)
def communicability_matrix_with_belta(graph: nx.Graph, belta: float, distance: int) -> list[list[float]]:
    """
    计算距离为3以内所有路径下源节点感染目标节点的总概率。
    相加得到最终感染概率矩阵 P,并将其概率矩阵命名为通信矩阵，其中节点自身通信力为1.
    Pij表示，节点i对节点j的通信力。
    param:distance 确定传播路径的距离大小，取值1，2，3，4，5

    1-beta ** 2表示距离为2的一条路径下节点j不被节点i感染的概率
    **num_path_2表示节点i到节点j距离为2的所有路径数量。只有
    所有路径下的j都不被i所感染，节点j才不被节点i所感染，所以节点j不被感染的概率为
    一条路径不被感染的概率 ** 路径数量，最后取反获得节点j被节点i感染的概率。
    """
    # 获取节点 i 到其他节点的最短路径
    shortest_paths_info = {}
    for node in graph.nodes():
        shortest_paths_info[node] = nx.shortest_path_length(graph, node)

    # 提取不同层级的邻居节点
    layer_1 = {}
    layer_2 = {}
    layer_3 = {}
    for node in graph.nodes():
        layer_1[node] = set(node for node, distance1 in shortest_paths_info[node].items() if distance1 == 1)
        layer_2[node] = set(node for node, distance2 in shortest_paths_info[node].items() if distance2 == 2)
        layer_3[node] = set(node for node, distance3 in shortest_paths_info[node].items() if distance3 == 3)
    degree = nodes_degree(graph)  # 获取所有节点的度

    if distance == 1:
        A = nx.to_numpy_array(graph)  # 图graph的邻接矩阵
        P = (belta * A)
        np.fill_diagonal(P, 1)

        return P

    elif distance == 2:
        A = nx.to_numpy_array(graph)  # 图graph的邻接矩阵
        A_2 = np.linalg.matrix_power(A, 2)
        np.fill_diagonal(A_2, 0)  # 主对角线元素为0
        P = (belta * A) + (1 - (1 - belta ** 2) ** A_2)
        np.fill_diagonal(P, 1)

        return P

    elif distance == 3:
        A = nx.to_numpy_array(graph)  # 图graph的邻接矩阵
        A_2 = np.linalg.matrix_power(A, 2)
        np.fill_diagonal(A_2, 0)  # 主对角线元素为0

        A_3 = A_2 @ A
        node_index = node_to_index(graph)  # dict{node:index}

        # 调整 A_3
        for node, i in node_index.items():
            for neighbor in graph.neighbors(node):
                j = node_index[neighbor]
                A_3[i][j] -= degree[neighbor]
                A_3[i][j] += 1

        np.fill_diagonal(A_3, 0)  # 将A_3主对角线元素置0方便后续计算
        P = (belta * A) + (1 - (1 - belta ** 2) ** A_2) + (1 - (1 - belta ** 3) ** A_3)
        np.fill_diagonal(P, 1)

        return P

    elif distance == 4:
        A = nx.to_numpy_array(graph)  # 图graph的邻接矩阵
        A_2 = np.linalg.matrix_power(A, 2)
        np.fill_diagonal(A_2, 0)  # 主对角线元素为0

        A_3 = A_2 @ A
        node_index = node_to_index(graph)  # dict{node:index}

        # 调整 A_3
        for node, i in node_index.items():
            for neighbor in graph.neighbors(node):
                j = node_index[neighbor]
                A_3[i][j] -= degree[neighbor]
                A_3[i][j] += 1

        np.fill_diagonal(A_3, 0)  # 将A_3主对角线元素置0方便后续计算

        A_4 = A_3 @ A
        pretreatment_nodes = {node: layer_1[node] & layer_2[node] for node in graph.nodes()}  # 计算预处理节点
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
        A = nx.to_numpy_array(graph)  # 图graph的邻接矩阵
        A_2 = np.linalg.matrix_power(A, 2)
        np.fill_diagonal(A_2, 0)  # 主对角线元素为0

        A_3 = A_2 @ A
        node_index = node_to_index(graph)  # dict{node:index}
        # 调整 A_3
        for node, i in node_index.items():
            for neighbor in graph.neighbors(node):
                j = node_index[neighbor]
                A_3[i][j] -= degree[neighbor]
                A_3[i][j] += 1
        np.fill_diagonal(A_3, 0)  # 将A_3主对角线元素置0方便后续计算

        A_4 = A_3 @ A
        pretreatment_nodes4 = {node: layer_1[node] & layer_2[node] for node in graph.nodes()}  # 计算预处理节点
        for node, i in node_index.items():
            for pending_node in pretreatment_nodes4[node]:
                j = node_index[pending_node]
                A_4[i][j] -= degree[pending_node]
                if pending_node in layer_1[node]:
                    A_4[i][j] += 1
        np.fill_diagonal(A_4, 0)  # 将A_4主对角线元素置0方便后续计算

        A_5 = A_4 @ A
        pretreatment_nodes5 = {node: layer_1[node] & layer_2[node] & layer_3[node] for node in graph.nodes()}  # 计算预处理节点
        for node, i in node_index.items():
            for pending_node in pretreatment_nodes5[node]:
                j = node_index[pending_node]
                A_5[i][j] -= degree[pending_node]
                if pending_node in layer_1[node]:
                    A_5[i][j] += 1

        np.fill_diagonal(A_5, 0)  # 将A_5主对角线元素置0方便后续计算
        P = (belta * A) + (1 - (1 - belta ** 2) ** A_2) + (1 - (1 - belta ** 3) ** A_3) + (
                    1 - (1 - belta ** 4) ** A_4) + (1 - (1 - belta ** 5) ** A_5)
        np.fill_diagonal(P, 1)

        return P
    else:
        print("Unselected distance")


def contribution_matrix(graph: nx.Graph, belta: float, distance=3) -> float:
    """
    计算考虑通信矩阵X的节点贡献度，并返回贡献矩阵
    假设贡献矩阵为C，则Cij表示节点j对节点i的贡献度。
    """
    selected_distance = distance
    node_index = node_to_index(graph)  # dict{'node':index}
    X = communicability_matrix_with_belta(graph, belta, selected_distance)  # 获取通信力矩阵X
    improved_valid_degree = valid_degree_with_self_degree(graph)  # dict{node:value}

    # 使用广播直接计算贡献矩阵
    C = X * np.array([improved_valid_degree[node] for node in node_index.keys()])

    return C


def valid_degree_promotion(graph: nx.Graph, belta: float, propagated_distance=3):
    """
    计算提升后的所有节点的有效度，考虑通信矩阵
    返回值：dict['node', 'value']
    """

    valid_degree_centrality = {}
    node_index = node_to_index(graph)  # dict{'node':index}
    C = contribution_matrix(graph, belta, distance=propagated_distance)  # 获取贡献度矩阵C
    for node, i in node_index.items():
        valid_degree_centrality[node] = round(np.sum(C[i, :]), 2)  # 节点node对应的索引i的行之和即是节点node的最终有效度

    return valid_degree_centrality


