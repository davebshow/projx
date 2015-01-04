# -*- coding: utf-8 -*-
import networkx as nx


def test_graph():
    """
    The first tests will use this function I assume.
    :returns: networkx.Graph
    """
    g = nx.Graph([
        (1, 2, {'type': 'works_at'}),
        (1, 3, {'type': 'lives_in'}),
        (2, 3, {'type': 'located_in'}),
        (3, 4, {'type': 'connected_to'}),
        (4, 5, {'type': 'connected_to'}),
        (10, 4, {'type': 'connected_to'}),
        (5, 6, {'type': 'lives_in'}),
        (7, 3, {'type': 'lives_in'}),
        (8, 5, {'type': 'works_at'}),
        (7, 2, {'type': 'works_at'}),
        (8, 4, {'type': 'lives_in'}),
        (7, 4, {'type': 'works_at'}),
        (9, 4, {'type': 'lives_in'}),
        (9, 10, {'type': 'works_at'}),
        (11, 3, {'type': 'lives_in'}),
        (12, 5, {'type': 'lives_in'})
    ])
    g.node[1] = {'type': 'Person', 'name': 'davebshow'}
    g.node[2] = {'type': 'Institution', 'name': 'western'}
    g.node[3] = {'type': 'City', 'name': 'london'}
    g.node[4] = {'type': 'Institution', 'name': 'the matrix'}
    g.node[5] = {'type': 'City', 'name': 'toronto'}
    g.node[6] = {'type': 'Person', 'name': 'gandalf'}
    g.node[7] = {'type': 'Person', 'name': 'versae'}
    g.node[8] = {'type': 'Person', 'name': 'neo'}
    g.node[9] = {'type': 'Person', 'name': 'r2d2'}
    g.node[10] = {'type': 'City', 'name': 'alderon'}
    g.node[11] = {'type': 'Person', 'name': 'curly'}
    g.node[12] = {'type': 'Person', 'name': 'adam'}
    return g


project_etl = {
    "extractor": {
        "networkx": {
            "class": "subgraph",
            "node_type_attr": "type",
            "edge_type_attr": "type",
            "traversal": [
                {"node": {"type": "Person", "alias": "p1"}},
                {"edge": {}},
                {"node": {"alias": "wild"}},
                {"edge": {}},
                {"node": {"type": "Person", "alias": "p2"}}
            ]
        }
    },
    "transformers": [
        {
            "project": {
                "method": {"jaccard": {"over": ["Institution", "City"]}},
                "pattern": [
                    {"node": {"alias": "p1"}},
                    {"edge": {}},
                    {"node": {"alias": "p2"}}
                ],
                "set": [
                    {
                        "alias": "NEW",
                        "key": "name",
                        "value":"",
                        "value_lookup": "wild.name"
                    }
                ],
                "delete": {"alias": ["wild"]}
            }
        }
    ],
    "loader": {
        "networkx": {}
    }
}


transfer_etl = {
    "extractor": {
        "networkx": {
            "class": "graph",
            "node_type_attr": "type",
            "edge_type_attr": "type",
            "traversal": [
                {"node": {"type": "City", "alias": "c"}},
                {"edge": {}},
                {"node": {"type": "Institution", "alias": "i"}},
            ]
        }
    },
    "transformers": [
        {
            "transfer": {
                "pattern": [
                    {"node": {"alias": "c"}},
                    {"edge": {}},
                    {"node": {"alias": "i"}}
                ],
                "set": [
                    {
                        "alias": "i",
                        "key": "city",
                        "value":"",
                        "value_lookup": "c.name"
                    }
                ],
                "delete": {"alias": ["c"]}
            }
        }
    ],
    "loader": {
        "networkx": {}
    }
}


combine_etl = {
    "extractor": {
        "networkx": {
            "class": "graph",
            "node_type_attr": "type",
            "edge_type_attr": "type",
            "traversal": [
                {"node": {"type": "City", "alias": "c"}},
                {"edge": {}},
                {"node": {"type": "Institution", "alias": "i"}},
            ]
        }
    },
    "transformers": [
        {
            "combine": {
                "pattern": [
                    {"node": {"alias": "c"}},
                    {"edge": {}},
                    {"node": {"alias": "i"}}
                ],
                "set": [
                    {
                        "alias": "NEW",
                        "key": "type",
                        "value":"GeoInst",
                        "value_lookup": ""
                    },
                    {
                        "alias": "NEW",
                        "key": "city_name",
                        "value":"",
                        "value_lookup": "c.name"
                    },
                    {
                        "alias": "NEW",
                        "key": "inst_name",
                        "value":"",
                        "value_lookup": "i.name"
                    }
                ],
                "delete": {"alias": ["c", "i"]}
            }
        }
    ],
    "loader": {
        "networkx": {}
    }
}


multi_etl = {
    "extractor": {
        "networkx": {
            "traversal": [
                {
                    "node": {
                        "alias": "i",
                        "type": "Institution"
                    }
                },
                {
                    "edge": {}
                },
                {
                    "node": {
                        "alias": "p1",
                        "type": "Person"
                    }
                },
                {
                    "edge": {}
                },
                {
                    "node": {
                        "alias": "c",
                        "type": "City"
                    }
                },
                {
                    "edge": {}
                },
                {
                    "node": {
                        "alias": "p2",
                        "type": "Person"
                    }
                }
            ],
            "type": "subgraph"
        }
    },
    "loader": {
        "networkx": {}
    },
    "transformers": [
        {
            "transfer": {
                "delete": {
                    "alias": []
                },
                "method": {
                    "none": {
                        "over": {}
                    }
                },
                "pattern": [
                    {
                        "node": {
                            "alias": "i"
                        }
                    },
                    {
                        "edge": {}
                    },
                    {
                        "node": {
                            "alias": "p1"
                        }
                    }
                ],
                "set": [
                    {
                        "alias": "p1",
                        "key": "inst",
                        "value_lookup": "i.name"
                    }
                ]
            }
        },
        {
            "project": {
                "delete": {
                    "alias": [
                        "c"
                    ]
                },
                "method": {
                    "jaccard": {
                        "over": [
                            "City"
                        ]
                    }
                },
                "pattern": [
                    {
                        "node": {
                            "alias": "p1"
                        }
                    },
                    {
                        "edge": {}
                    },
                    {
                        "node": {
                            "alias": "p2"
                        }
                    }
                ],
                "set": []
            }
        }
    ]
}

def draw_simple_graph(graph, node_type_attr='type',
                      edge_label_attr='weight', show_edge_labels=True,
                      label_attrs=['name']):
    """
    Utility function to draw a labeled, colored graph with Matplotlib.

    :param graph: networkx.Graph    
    """
    lbls = labels(graph, label_attrs=label_attrs)
    clrs = colors(graph, node_type_attr=node_type_attr)
    pos = nx.spring_layout(graph, weight=None)
    if show_edge_labels:
        e_labels = edge_labels(graph, edge_label_attr=edge_label_attr)
    else:
        e_labels = {}
    nx.draw_networkx(graph, pos=pos, node_color=clrs)
    nx.draw_networkx_edge_labels(graph, pos=pos, edge_labels=e_labels)
    nx.draw_networkx_labels(graph, pos=pos, labels=lbls)


def labels(graph, label_attrs=['name']):
    """
    Utility function that aggreates node attributes as
    labels for drawing graph in Ipython Notebook.

    :param graph: networkx.Graph
    :returns: Dict. Nodes as keys, labels as values.
    """
    labels_dict = {}
    for node, attrs in graph.nodes(data=True):
        label = u''
        for k, v in attrs.items():
            if k in label_attrs:
                try:
                    label += u'{0}: {1}\n'.format(k, v)
                except:
                    label += u'{0}: {1}\n'.format(k, v).encode('utf-8')
        labels_dict[node] = label
    return labels_dict


def edge_labels(graph, edge_label_attr='weight'):
    """
    Utility function that aggreates node attributes as
    labels for drawing graph in Ipython Notebook.

    :param graph: networkx.Graph
    :returns: Dict. Nodes as keys, labels as values.
    """
    labels_dict = {}
    for i, j, attrs in graph.edges(data=True):
        label = attrs.get(edge_label_attr, '')
        labels_dict[(i, j)] = label
    return labels_dict


def colors(graph, node_type_attr='type'):
    """
    Utility function that generates colors for node
    types for drawing graph in Ipython Notebook.

    :param graph: networkx.Graph
    :returns: Dict. Nodes as keys, colors as values.
    """
    colors_dict = {}
    colors = []
    counter = 1
    for node, attrs in graph.nodes(data=True):
        if attrs[node_type_attr] not in colors_dict:
            colors_dict[attrs[node_type_attr]] = float(counter)
            colors.append(float(counter))
            counter += 1
        else:
            colors.append(colors_dict[attrs[node_type_attr]])
    return colors


def remove_edges(g, min_weight):
    for edge in g.edges(data=True):
        if edge[2]['weight'] < min_weight:
            g.remove_edge(edge[0], edge[1])
    for node, deg in g.degree().items():
        if deg == 0:
            g.remove_node(node)
    return g


def proj_density(g, start_val, interval, num_proj):
    dens = []
    cutoffs = []
    for i in range(num_proj):
        proj = remove_edges(g.copy(), start_val)
        dens.append(nx.density(proj))
        cutoffs.append(start_val)
        start_val += interval
    return cutoffs, dens
