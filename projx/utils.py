# -*- coding: utf-8 -*-
"""
These are mainly for testing/demoing the library.
"""
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
        (12, 5, {'type': 'lives_in'}),
        (12, 13, {'type': 'works_at'}),
        (13, 5, {'type': 'located_in'}),
        (13, 14, {'type': 'works_at'})
    ])
    g.node[1] = {'type': 'Person', 'label': 'davebshow'}
    g.node[2] = {'type': 'Institution', 'label': 'western'}
    g.node[3] = {'type': 'City', 'label': 'london'}
    g.node[4] = {'type': 'Institution', 'label': 'the matrix'}
    g.node[5] = {'type': 'City', 'label': 'toronto'}
    g.node[6] = {'type': 'Person', 'label': 'gandalf'}
    g.node[7] = {'type': 'Person', 'label': 'versae'}
    g.node[8] = {'type': 'Person', 'label': 'neo'}
    g.node[9] = {'type': 'Person', 'label': 'r2d2'}
    g.node[10] = {'type': 'City', 'label': 'alderon'}
    g.node[11] = {'type': 'Person', 'label': 'curly'}
    g.node[12] = {'type': 'Person', 'label': 'adam'}
    g.node[13] = {'type': 'Institution', 'label': 'canland'}
    g.node[14] = {'type': 'Person', 'label': 'bro'}
    return g



project_etl = {
    "extractor": {
        "networkx": {
            "type": "subgraph",
            "node_type_attr": "type",
            "edge_type_attr": "type",
            "traversal": [
                {"node": { "alias": "p1", "type": "Person"}},
                {"edge": {}},
                {"node": {"alias": "wild"}},
                {"edge": {}},
                {"node": {"alias": "p2", "type": "Person"}}
            ]
        }
    },
    "transformers": [
        {
            "project": {
                "pattern": [
                    {"node": {"alias": "p1"}},
                    {"edge": {}},
                    {"node": {"alias": "p2"}}
                ],
                "set": [
                    {"key": "name", "value_lookup": "wild.label"}
                ],
                "method": {
                    "jaccard": {
                        "args": ["Institution", "City"]
                    }
                },
                "delete": {
                    "alias": ["wild"]
                }
            }
        }
    ],
    "loader": {
        "nx2nx": {}
    }
}


transfer_etl = {
    "extractor": {
        "networkx": {
            "type": "graph",
            "node_type_attr": "type",
            "edge_type_attr": "type",
            "traversal": [
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
                        "alias": "i",
                        "type": "Institution"
                    }
                }
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
                    {"key": "city", "value_lookup": "c.label"}
                ],
                "method": {
                    "edges": {
                        "args": ["Person"]
                    }
                },
                "delete": {
                    "alias": ["c"]
                }
            }
        }
    ],
    "loader": {
        "nx2nx": {}
    }
}


combine_etl = {
    "extractor": {
        "networkx": {
            "type": "graph",
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
                        "key": "type",
                        "value":"GeoInst",
                        "value_lookup": ""
                    },
                    {
                        "key": "city_name",
                        "value":"",
                        "value_lookup": "c.label"
                    },
                    {
                        "key": "inst_name",
                        "value":"",
                        "value_lookup": "i.label"
                    }
                ],
                "delete": {"alias": ["c", "i"]}
            }
        }
    ],
    "loader": {
        "nx2nx": {}
    }
}


multi_transform_etl = {
    "extractor": {
        "networkx": {
            "traversal": [
                {"node": {"alias": "p1", "type": "Person"}},
                {"edge": {}},
                {"node": {"alias": "i", "type": "Institution"}},
                {"edge": {}},
                {"node": {"alias": "c", "type": "City"}}
            ],
            "type": "subgraph"
        }
    },
    "loader": {
        "nx2nx": {}
    },
    "transformers": [
        {
            "transfer": {
                "delete": {
                    "alias": []
                },
                "method": {
                    "attrs": {
                        "args": []
                    }
                },
                "pattern": [
                    {"node": {"alias": "i"}},
                    {"edge": {}},
                    {"node": {"alias": "c"}}
                ],
                "set": [
                    {"key": "inst", "value_lookup": "i.label"}
                ]
            }
        },
        {
            "project": {
                "delete": {
                    "alias": [
                        "i"
                    ]
                },
                "method": {
                    "jaccard": {
                        "args": [
                            "Institution"
                        ]
                    }
                },
                "pattern": [
                    {"node": {"alias": "p1"}},
                    {"edge": {}},
                    {"node": {"alias": "c"}}
                ],
                "set": [{}]
            }
        }
    ]
}


neo4j2nx_etl = {
    "extractor": {
        "neo4j": {
            "query": "match (n)--(r:Recipe)--(m) return n, r, m",
            "uri": "http://localhost:7474/db/data/"
        }

    },
    "transformers": [
        {
            "node": {
                "pattern": [{"node": {"alias": "n", "unique": "UniqueId"}}],
                "set": [
                    {"key": "name", "value_lookup": "n.UniqueId"},
                    {"key": "type", "value": "Ingredient"}
                ]
            },
        },
        {
            "node": {
                "pattern": [{"node": {"alias": "m", "unique": "UniqueId"}}],
                "set": [
                    {"key": "name", "value_lookup": "m.UniqueId"},
                    {"key": "type", "value": "Ingredient"}
                ]
            },
        },
        {
            "edge": {
                "pattern": [
                    {"node": {"alias": "n", "unique": "UniqueId"}},
                    {"edge": {}},
                    {"node": {"alias": "m", "unique": "UniqueId"}}
                ],
                "set": [
                    {"key": "name", "value_lookup": "r.UniqueId"}
                ],
            }
        }
    ],
    "loader": {
        "neo4j2nx": {}
    }
}


neo4j2edgelist_etl = {
    "extractor": {
        "neo4j": {
            "query": "match (n)--(r:Recipe)--(m) return n, r, m"
        }
    },
    "transformers": [
        {
            "node": {
                "pattern": [{"node": {"alias": "n", "unique": "UniqueId"}}],
                "set": [
                    {"key": "name", "value_lookup": "n.UniqueId"},
                    {"key": "type", "value": "Ingredient"}
                ]
            },
        },
        {
            "node": {
                "pattern": [{"node": {"alias": "m", "unique": "UniqueId"}}],
                "set": [
                    {"key": "name", "value_lookup": "m.UniqueId"},
                    {"key": "type", "value": "Ingredient"}
                ]
            },
        },
        {
            "edge": {
                "pattern": [
                    {"node": {"alias": "n", "unique": "UniqueId"}},
                    {"edge": {}},
                    {"node": {"alias": "m", "unique": "UniqueId"}}
                ],
                "set": [
                    {"key": "name", "value_lookup": "r.UniqueId"}
                ],
            }
        }
    ],
    "loader": {
        "neo4j2edgelist": {"delim": ",", "filename": "demo.csv", "newline": "\n"}
    }
}


edgelist2neo4j_etl = {
    "extractor": {
        "edgelist": {
            "filename": "data/flickr-groupmemberships/out.flickr-groupmemberships",
            "delim": " ",
            "pattern": [
                {"node": {"alias": "n"}},
                {"edge": {}},
                {"node": {"alias": "m"}}
            ]
        }
    },
    "transformers": [
        {
            "edge": {
                "pattern": [
                    {"node": {"alias": "n", "label": "User"}},
                    {"edge": {"label": "IN"}},
                    {"node": {"alias": "m", "label": "Group"}}
                ]
            }
        }
    ],
    "loader": {
        "edgelist2neo4j": {
            "uri": "http://localhost:7474/db/data",
            "stmt_per_req": 500,
            "req_per_tx": 25,
            "indicies": [
                {"label": "User", "attr": "UniqueId"},
                {"label": "Group", "attr": "UniqueId"}
            ]
        }
    }
}


def draw_simple_graph(graph, node_type_attr='type',
                      edge_label_attr='weight', show_edge_labels=True,
                      label_attrs=['label']):
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


def labels(graph, label_attrs=['label']):
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
