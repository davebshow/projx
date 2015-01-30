# -*- coding: utf-8 -*-
"""
These are the core functions/classes for interacting with NetworkX.
"""
from itertools import chain
import networkx as nx


def reset_index(graph):
    """
    This clobbers your nodes, best protect them.

    :param graph: networx.Graph
    :returns: networkx.Graph
    """
    mapping = dict(zip(graph.nodes(), range(0, graph.number_of_nodes())))
    # Change nodes to integers.
    graph = nx.relabel_nodes(graph, mapping)
    return graph


def match(node_type_seq, edge_type_seq, graph, node_alias=None,
          node_type_attr="type", edge_type_attr="type"):
    """
    Executes traversals to perform initial match on pattern.

    :param graph: networkx.Graph
    :returns: List of lists. The matched paths.
    """
    start_type = node_type_seq[0]
    path_list = []
    for node, attrs in graph.nodes(data=True):
        if attrs[node_type_attr] == start_type or not start_type:
            paths = traverse(node, node_type_seq[1:], edge_type_seq, graph,
                             node_alias, node_type_attr, edge_type_attr)
            path_list.append(paths)
    paths = chain.from_iterable(path_list)
    return paths


def project(source, target, graph, method="jaccard", params=None, attrs=None,
            node_type_attr="type", edge_type_attr="type"):
    """
    Executes graph "PROJECT" projection.

    :param source: Int. Source node for transformation.
    :param target: Int. Target node for transformation.
    :param attrs: Dict. Attrs to be set during transformation.
    :param graph: networkx.Graph. Graph of subgraph to transform.
    :returns: networkx.Graph. A projected copy of the wrapped graph
    or its subgraph.
    """
    if params is None:
        params = []
    if attrs is None:
        attrs = {}
    if method in ["jaccard", "newman"]:
        snbrs = {node for node in graph[source].keys()
                 if graph.node[node][node_type_attr] in params}
        tnbrs = {node for node in graph[target].keys()
                 if graph.node[node][node_type_attr] in params}
        intersect = snbrs & tnbrs
        if method == "jaccard":
            union = snbrs | tnbrs
            weight = float(len(intersect)) / len(union)
        elif method == "newman":
            weight = sum([1.0 / (len(graph[n]) - 1) for n in intersect
                          if len(graph[n]) > 1])
        attrs["weight"] = weight
    if graph.has_edge(source, target):
        edge_attrs = graph[source][target]
        merged_attrs = merge_attrs(attrs, edge_attrs,
                                   [edge_type_attr, "weight", "label"])
        graph.adj[source][target] = merged_attrs
        graph.adj[target][source] = merged_attrs
    else:
        graph.add_edge(source, target, attrs)
    return graph


def transfer(source, target, graph, method="edges", params=None, attrs=None,
             node_type_attr="type", edge_type_attr="type", **kwargs):
    """
    Execute a graph "TRANSFER" projection.

    :param source: Int. Source node for transformation.
    :param target: Int. Target node for transformation.
    :param attrs: Dict. Attrs to be set during transformation.
    :param graph: networkx.Graph. Graph of subgraph to transform.
    :returns: networkx.Graph. A projected copy of the wrapped graph
    or its subgraph.
    """
    if params is None:
        params = []
    if attrs is None:
        attrs = {}
    if method == "edges":
        nbrs = {k: v for (k, v) in graph[source].items()
                if graph.node[k][node_type_attr] in params}
        edges = zip([target] * len(nbrs), nbrs,
                    [v for (k, v) in nbrs.items()])
        graph = _add_edges_from(graph, edges)
    old_attrs = graph.node[target]
    merged_attrs = merge_attrs(attrs, old_attrs,
                               [node_type_attr, "label", "role"])
    graph.node[target] = merged_attrs
    return graph


def combine(source, target, graph, node_id="", attrs=None,
            node_type_attr="type", edge_type_attr="type"):

    """
    Executes graph "COMBINE" projection.

    :param source: Int. Source node for transformation.
    :param target: Int. Target node for transformation.
    :param attrs: Dict. Attrs to be set during transformation.
    :param graph: networkx.Graph. Graph of subgraph to transform.
    :param node_id: Int. Id for new node, will autoassign, but
    :returns: networkx.Graph. A projected copy of the wrapped graph
    or its subgraph.
    """
    if attrs is None:
        attrs = {}
    if not node_id:
        try:
            node_id = max(graph.nodes())
        except:
            raise Exception("Please specify a kwarg 'node_id'")
    node_type = attrs.get(node_type_attr, "")
    if not node_type:
        node_type = "{0}_{1}".format(
            graph.node[source][node_type_attr],
            graph.node[target][node_type_attr]
        )
        attrs[node_type_attr] = node_type
    graph.add_node(node_id, attrs)
    nbrs = dict(graph[source])
    nbrs.update(dict(graph[target]))
    nbrs = {k: v for (k, v) in nbrs.items()
            if graph.node[k][node_type_attr] != node_type}
    edges = zip([node_id] * len(nbrs), nbrs,
                [v for (_, v) in nbrs.items()])
    graph = _add_edges_from(graph, edges)
    return graph


def traverse(start, node_type_seq, edge_type_seq, graph,
             node_alias=None, node_type_attr="type", edge_type_attr="type"):
    """
    This is a controlled depth, depth first traversal of a NetworkX
    graph and the core of this library. Criteria for searching depends
    on a start node and a sequence of types as designated by the node/edge
    type seq. It does not allow cycles or backtracking. Could be very memory
    inefficient in very dense graph with 3 + type queries.

    :param start: Integer. Starting point for the traversal.
    :param node_type_seq: List of strings. Derived from the match pattern.
    :param node_type_seq: List of strings. Derived from the match pattern.
    :param graph: networkx.Graph
    :returns: List of lists. All matched paths.
    """
    # Initialize a stack to keep
    # track of traversal progress.
    stack = [start]
    # Store all valid paths based on type sequence.
    paths = []
    # Keep track of traversal moves to avoid cycles.
    visited_from = {}
    # The traversal will begin at the designated start point.
    current = start
    # Track depth from start node to watch for successful sequence match.
    depth = 0
    # This is the len of a successful sequence.
    max_depth = len(node_type_seq)
    # When the stack runs out, all candidate nodes have been visited.
    while len(stack) > 0:
        # Traverse!
        if depth < max_depth:
            nbrs = set(graph[current]) - set([current])
            for nbr in nbrs:
                edge_type = graph[current][nbr].get(
                    edge_type_attr,
                    None
                )
                attrs = graph.node[nbr]
                # Here check candidate node validity.
                # Make sure this path hasn"t been checked already.
                # Make sure it matches the type sequence.
                # Make sure it"s not backtracking on same path.
                visited_from.setdefault(nbr, [])
                if (current not in visited_from[nbr] and
                        nbr not in stack and
                        (edge_type == edge_type_seq[depth] or
                         edge_type_seq[depth] == "") and
                        (attrs[node_type_attr] == node_type_seq[depth]
                         or node_type_seq[depth] == "")):
                    visited_from[nbr].append(current)
                    # Continue traversal at next depth.
                    current = nbr
                    stack.append(current)
                    depth += 1
                    break
            # If no valid nodes are available from
            # this position, backtrack.
            else:
                stack.pop()
                if len(stack) > 0:
                    current = stack[-1]
                    depth -= 1
        # If max depth reached, store the
        # valid node sequence.
        else:
            path = list(stack)
            if node_alias:
                path = Record(path, node_alias)
            paths.append(path)
            # Backtrack and keep checking.
            stack.pop()
            current = stack[-1]
            depth -= 1
    return paths


def build_subgraph(paths, graph, records=False):
    """
    Takes the paths returned by match and builds a graph.
    :param paths: List of lists.
    :returns: networkx.Graph. Matched sugraph.
    """
    g = nx.Graph()
    for path in paths:
        if records:
            path = path._list
        combined_paths = _combine_paths(path)
        for edges in combined_paths:
            attrs = graph[edges[0]][edges[1]]
            g.add_edge(edges[0], edges[1], attrs)
    for node in g.nodes():
        g.node[node] = dict(graph.node[node])
    return g


def merge_attrs(new_attrs, old_attrs, reserved=[]):
    """
    Merges attributes counting repeated attrs with dicts.
    Kind of ugly, will need to take a look at this.
    :param new_attrs: Dict.
    :param old_attrs: Dict.
    :reserved: List. A list of attributes that cannot have more than value.
    :returns: Dict.
    """
    attrs = {}
    for k, v in old_attrs.items():
        if k in reserved:
            attrs[k] = v
        elif isinstance(v, dict):
            attrs[k] = dict(v)
        elif isinstance(v, str) or isinstance(v, unicode):
            attrs[k] = {v: 1}
    for k, v in new_attrs.items():
        if k in reserved:
            attrs[k] = v
        elif k in attrs:
            count_dict = attrs[k]
            if isinstance(v, dict):
                for i, j in v.items():
                    count_dict.setdefault(i, 0)
                    count_dict[i] += j
            elif isinstance(v, str) or isinstance(v, unicode):
                count_dict.setdefault(v, 0)
                count_dict[v] += 1
            attrs[k] = count_dict
        else:
            if isinstance(v, dict):
                attrs[k] = dict(v)
            elif isinstance(v, str) or isinstance(v, unicode):
                attrs[k] = {v: 1}
    return attrs


class NXProjector(object):
    def __init__(self, id_counter):
        """
        This class holds the info and methods necessary for performing the ETL
        actions on a networkx.Graph. It is not a wrapper, and does not store
        the actual graph, just operates on it..

        :param id_counter: Int. Used to handle combine ids.

        """
        self._id_counter = id_counter
        self._transformation = {}
        self._transformation_init()

    def transformation_wrapper(self, verb):
        """
        Wraps the transformation methods and adds them to the transformations
        dictionary.

        :param verb: Str. The ProjX verb assiociated with the wrapped
        function.
        """
        def wrapper(fn):
            self._transformation[verb] = fn
        return wrapper

    def _get_transformation(self):
        """
        Return transformation for transformation property.

        :returns: Dict. A dict containing a mapping of verbs to transformation
            methods.
        """
        return self._transformation
    transformations = property(fget=_get_transformation)

    def _transformation_init(self):
        """
        A series of functions representing transformations. These are
        wrapped by the transformation wrapper and added to the transformations
        dict. Later during the parsing and execution phase these are called as
        pointers to the various graph transformation methods
        (transfer and project).
        """
        @self.transformation_wrapper("project")
        def execute_project(source, target, graph, attrs, node_type_attr,
                            edge_type_attr, **kwargs):
            method = kwargs.get("method", {})
            params = kwargs.get("params", [])
            return project(source, target, graph, method, params, attrs,
                           node_type_attr, edge_type_attr)

        @self.transformation_wrapper("transfer")
        def execute_transfer(source, target, graph, attrs, node_type_attr,
                             edge_type_attr, **kwargs):
            method = kwargs.get("method", {})
            params = kwargs.get("params", [])
            return transfer(source, target, graph, method, params, attrs,
                            node_type_attr, edge_type_attr)

        @self.transformation_wrapper("combine")
        def execute_combine(source, target, graph, attrs, node_type_attr,
                            edge_type_attr, **kwargs):
            self._id_counter += 1
            node_id = int(self._id_counter)
            return combine(source, target, graph, node_id, attrs,
                           node_type_attr, edge_type_attr)


class Record(object):

    def __init__(self, path, alias):
        self._list = path
        self._dict = {}
        for i in range(len(path)):
            self._dict[alias[i]] = path[i]

    def __getitem__(self, item):
        if isinstance(item, str):
            return self._dict[item]
        elif isinstance(item, int):
            return self._list[item]
        else:
            raise Exception("Bad index.")


def _add_edges_from(graph, edges, edge_type_attr="type"):
    """
    An alternative to the networkx.Graph.add_edges_from.
    Handles non-reserved attributes as sets.

    :param graph: networkx.Graph
    :param edges: List of tuples. Tuple contains two node ids Int and an
    attr Dict.
    """
    for source, target, attrs in edges:
        if graph.has_edge(source, target):
            edge_attrs = graph[source][target]
            merged_attrs = merge_attrs(attrs, edge_attrs,
                                        [edge_type_attr, "weight", "label"])
            graph.adj[source][target] = merged_attrs
            graph.adj[target][source] = merged_attrs
        else:
            graph.add_edge(source, target, attrs)
    return graph


def _combine_paths(path):
    """
    Turn path list into edge list.
    :param path: List. A list of nodes representing a path.
    :returns: List. A list of edge tuples.
    """
    edges = []
    for i, node in enumerate(path[1:]):
        edges.append((path[i], node))
    return edges
