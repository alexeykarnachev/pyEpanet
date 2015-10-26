import networkx as nx

"""
Common graph processing routines.
"""

__author__ = 'Alexey Karnachev'


def find_nodes(g, comparison_func):
    """
    Function to find specific nodes due to comparison function.
    example:
    junctions = NetworkDataModel.find_nodes(lambda x: (x['TYPE'] == 'JUNCTION'))
    """
    matched_nodes = []
    for n in g.nodes_iter():
        node = g.node[n]
        if comparison_func(node):
            matched_nodes.append(n)
    return matched_nodes


def find_edges(g, comparison_func):
    """
    Function to find specific edges due to comparison function.
    example:
    valves = NetworkDataModel.find_edges(lambda x: (x['TYPE'] == 'TCV') and float(x['SETTING']) > 1)
    """
    matched_edges = []
    for n1, n2 in g.edges_iter():
        edge = g.edge[n1][n2]
        if comparison_func(edge):
            matched_edges.append((n1, n2))
    return matched_edges


def change_nodes_attr(g, nodes, attrs_dict):
    """
    Procedure to change nodes attributes due to attributes dictionary
    example:
    NetworkDataModel.change_nodes_attr(junctions, {'PATTERN': 'demand_pattern'})
    """
    for n in nodes:
        for attr in attrs_dict:
            value = attrs_dict[attr]
            g.node[n][attr] = value


def change_graph_attr(g, attrs_dict):
    """
    Procedure to change graph attributes due to attributes dictionary
    example:
    NetworkDataModel.change_graph_attr({'TITLE': 'New Title'})
    """
    for attr in attrs_dict:
        value = attrs_dict[attr]
        g.graph[attr] = value


def change_edges_attr(g, edges, attrs_dict):
    """
    Procedure to change nodes attributes due to attributes dictionary
    example:
    NetworkDataModel.change_edges_attr(edges, {'ROUGHNESS': '10'})
    """
    for n1, n2 in edges:
        for attr in attrs_dict:
            value = attrs_dict[attr]
            g.edge[n1][n2][attr] = value


def remove_unconnected_nodes(g):
    """
    This procedure removes all unconnected nodes from graph.
    """
    nodes_to_remove = []
    for n in g.nodes_iter():
        neighbors = g.neighbors(n)
        if len(neighbors) == 0:
            nodes_to_remove.append(n)
    g.remove_nodes_from(nodes_to_remove)


def remove_leaves(g, equal_attrs_dict, merging_funcs_dict):
    """
    This procedure remove leaves from graph.
    equal_attrs_dict: dictionary with nodes attributes and attributes values, which must be equal to merge
                        neighbor and leave.
    merging_funcs_dict: dictionary with nodes attributes, which must be merged, and values,
                        which are rules(functions), to merge attributes

    example:
        import networkx as nx
        import matplotlib.pyplot as plt

        G = nx.Graph()
        G.add_node('node1', {'demand': 100, 'high': 10, 'type': '1'})
        G.add_node('node2', {'demand': 70, 'high': 30, 'type': '1'})
        G.add_node('node3', {'demand': 30, 'high': 20, 'type': '1'})
        G.add_node('node4', {'demand': 200, 'high': 80, 'type': '1'})
        G.add_node('node5', {'demand': 160, 'high': 80, 'type': '2'})
        G.add_edge('node1', 'node2')
        G.add_edge('node2', 'node3')
        G.add_edge('node3', 'node1')
        G.add_edge('node1', 'node4')
        G.add_edge('node2', 'node5')
        print(G.node)

        nx.draw(G, node_size=25)
        plt.show()

        N = NetworkDataModel()
        N.set_network(G)

        N.remove_leaves(G, {'type': '1'}, {'demand': (lambda a, b: a+b)})
        print(G.node)

        nx.draw(G, node_size=25)
        plt.show()
    """

    finished = False

    while not finished:
        nodes_to_remove = []
        for n in g.nodes_iter():
            neighbors = g.neighbors(n)
            if len(neighbors) == 1:

                can_be_merged = True
                neighbor = g.node[neighbors[0]]
                node = g.node[n]

                for attr in equal_attrs_dict.keys():
                    if (attr not in node) or (attr not in neighbor) or (neighbor[attr] != node[attr]):
                        can_be_merged = False

                if can_be_merged:
                    for attr in merging_funcs_dict:
                        func = merging_funcs_dict[attr]
                        neighbor[attr] = func(neighbor[attr], node[attr])
                    nodes_to_remove.append(n)

        if len(nodes_to_remove) == 0:
            finished = True
        else:
            g.remove_nodes_from(nodes_to_remove)


def set_middle_node(g,
                    new_node_dict,
                    target_edge_tuple,
                    attrs_change_functions_edge_1,
                    attrs_change_functions_edge_2):
    """
    This procedure sets the middle point on the specific edge. This edge will be replaced with two others edges.
    :param new_node_dict: Dictionary with new node parameters.
    :param target_edge_tuple: A tuple with target edge.
    :param attrs_change_functions_edge_1: A dictionary with functions, which will be applied to the original edge
    attributes.
    :param attrs_change_functions_edge_2: A dictionary with functions, which will be applied to the original edge
    attributes.

    example:
    import networkx as nx
    import matplotlib.pyplot as plt

    G = nx.Graph()
    G.add_node('node1', {'demand': 100, 'high': 10, 'type': '1'})
    G.add_node('node2', {'demand': 70, 'high': 30, 'type': '1'})
    G.add_node('node3', {'demand': 30, 'high': 20, 'type': '1'})
    G.add_node('node4', {'demand': 200, 'high': 80, 'type': '1'})
    G.add_node('node5', {'demand': 160, 'high': 80, 'type': '2'})
    G.add_edge('node1', 'node2', {'length': 1000, 'id': '1'})
    G.add_edge('node2', 'node3', {'length': 1000, 'id': '1'})
    G.add_edge('node3', 'node1', {'length': 1000, 'id': '1'})
    G.add_edge('node1', 'node4', {'length': 1000, 'id': '1'})
    G.add_edge('node2', 'node5', {'length': 1000, 'id': '1'})
    G.graph['COORDINATES'] = {'node1': (3, 2), 'node2': (4, 5), 'node3': (7, 5), 'node4': (7, 2), 'node5': (2, 7)}

    nx.draw(G, node_size=25, pos=G.graph['COORDINATES'])
    plt.show()

    N = NetworkDataModel()
    N.set_network(G)

    new_node_dict = {'demand': 0, 'high': 20, 'type': 3}
    target_edge_tuple = ('node1', 'node2')
    attrs_change_functions_edge_1 = {'length': lambda x: x / 2, 'id': lambda x: x + '_first'}
    attrs_change_functions_edge_2 = {'length': lambda x: x / 2, 'id': lambda x: x + '_second'}
    N.set_middle_node(new_node_dict, target_edge_tuple, attrs_change_functions_edge_1, attrs_change_functions_edge_2)

    new_x = (N.Network.graph['COORDINATES']['node1'][0] + N.Network.graph['COORDINATES']['node2'][0]) / 2
    new_y = (N.Network.graph['COORDINATES']['node1'][1] + N.Network.graph['COORDINATES']['node2'][1]) / 2
    N.Network.graph['COORDINATES']['node1|node2'] = (new_x, new_y)

    nx.draw(N.Network, node_size=25, pos=N.Network.graph['COORDINATES'])
    plt.show()
    """

    old_edge_dict = g.edge[target_edge_tuple[0]][target_edge_tuple[1]].copy()
    g.remove_edge(target_edge_tuple[0], target_edge_tuple[1])

    new_node_name = target_edge_tuple[0] + '_' + target_edge_tuple[1]
    g.add_node(new_node_name, new_node_dict)

    new_dict_1 = old_edge_dict.copy()
    for attr_to_change in attrs_change_functions_edge_1.keys():
        func = attrs_change_functions_edge_1[attr_to_change]
        new_dict_1[attr_to_change] = func(new_dict_1[attr_to_change])

    new_dict_2 = old_edge_dict.copy()
    for attr_to_change in attrs_change_functions_edge_2.keys():
        func = attrs_change_functions_edge_2[attr_to_change]
        new_dict_2[attr_to_change] = func(new_dict_2[attr_to_change])

    g.add_edge(target_edge_tuple[0], new_node_name, new_dict_1)
    g.add_edge(target_edge_tuple[1], new_node_name, new_dict_2)


def import_network_from_data_frames(nodes_df, edges_df, connections_df):
    """
    @param nodes_df:
    @param edges_df:
    @param connections_df:
    @return:
    """
    g = nx.Graph()

    g.add_nodes_from(nodes_df.index.tolist())
    for node in g.nodes_iter():
        attrs = nodes_df.loc[node].to_dict()
        g.node[node] = attrs

    for row in connections_df.iterrows():
        id_ = row[0]
        attrs = edges_df.loc[id_].to_dict()
        attrs['ID'] = id_
        edge = tuple(row[1])
        g.add_edge(edge[0], edge[1], attrs)

    return g
