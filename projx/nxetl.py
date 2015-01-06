# -*- coding: utf-8 -*-
from functools import partial
from itertools import chain
import networkx as nx


# NetworkX Module.
def nx2nx_loader(transformers, extractor, graph):
    """
    Loader for NetworkX graph.

    :param projector: projx.NXprojector
    :param paths: List of lists.
    :returns: networkx.Graph
    """
    context = extractor(graph)
    if len(transformers) > 1:
        graph = nx_transformer_pipeline(
            transformers,
            context["graph"],
            context["paths"],
            context["node_alias"],
            context["node_type_attr"],
            context["edge_type_attr"]
        )
    elif len(transformers) == 1:
        graph = nx_transformer(
            transformers,
            context["graph"],
            context["paths"],
            context["node_alias"],
            context["node_type_attr"],
            context["edge_type_attr"]
        )
    else:
        graph = context["graph"]
    return graph


def nx_transformer(transformers, graph, paths, node_alias, node_type_attr,
                   edge_type_attr):
    """
    Static transformer for NetworkX graph. Single transformation.

    :param projector: projx.NXprojector
    :param graph: networkx.Graph
    :param paths: List of lists.
    :returns: networkx.Graph
    """
    removals = set()
    projector = NXProjector(max(graph.nodes()))
    transformer = transformers[0]
    trans_kwrd = transformer.keys()[0]
    trans = transformer[trans_kwrd]
    pattern = trans["pattern"]
    source, target = _get_source_target(node_alias, pattern)
    to_set = trans.get("set", [])
    fn = projector.transformations[trans_kwrd]
    delete_alias = trans.get("delete", {}).get("alias", [])
    to_delete = [node_alias[alias] for alias in delete_alias]
    method = trans.get("method", {})
    for path in paths:
        source_node = path[source]
        target_node = path[target]
        attrs = _lookup_attrs(node_alias, graph, to_set, path)
        graph = fn(source_node, target_node, graph, attrs, node_type_attr,
                   edge_type_attr, method=method)
        for i in to_delete:
            removals.update([path[i]])
    graph.remove_nodes_from(removals)
    return graph


def nx_transformer_pipeline(transformers, graph, paths, node_alias,
                            node_type_attr, edge_type_attr):
    """
    Pipeline transformer for NetworkX graph. Multiple transformations.

    :param transformers: List.
    :param projector: projx.NXprojector
    :param graph: networkx.Graph
    :param paths: List of lists.
    :returns: networkx.Graph
    """
    removals = set()
    projector = NXProjector(max(graph.nodes()))
    for path in paths:
        for transformer in transformers:
            trans_kwrd = transformer.keys()[0]
            trans = transformer[trans_kwrd]
            pattern = trans["pattern"]
            source, target = _get_source_target(node_alias, pattern)
            source_node = path[source]
            target_node = path[target]
            to_set = trans.get("set", [])
            method = trans.get("method", {})
            attrs = _lookup_attrs(node_alias, graph, to_set, path)
            fn = projector.transformations[trans_kwrd]
            graph = fn(source_node, target_node, graph, attrs, node_type_attr,
                       edge_type_attr, method=method)
            delete_alias = trans.get("delete", {}).get("alias", [])
            to_delete = [node_alias[alias] for alias in delete_alias]
            for i in to_delete:
                removals.update([path[i]])
    graph.remove_nodes_from(removals)
    return graph


def nx_extractor(extractor, graph):
    """
    Parses etl extractor JSON and produces all of the data necessary for
    transformation.

    :param extractor: JSON.
    :param graph: networkx.Graph
    :returns: Dict.
    """
    proj_type = extractor.get("type", "subgraph")
    traversal = extractor.get("traversal", [])
    nodes = traversal[0::2]
    edges = traversal[1::2]
    node_type_attr = extractor.get("node_type_attr", "type")
    edge_type_attr = extractor.get("edge_type_attr", "type")
    try:
        node_alias = {node["node"]["alias"]: i for
                      (i, node) in enumerate(nodes)}
        edge_alias = {edge["edge"].get("alias", i): i for
                      (i, edge) in enumerate(edges)}
        node_type_seq = [node["node"].get(node_type_attr, "")
                         for node in nodes]
        edge_type_seq = [edge["edge"].get(edge_type_attr, "")
                         for edge in edges]
    except KeyError:
        raise Exception("Please define valid traversal sequence")
    graph = reset_index(graph)
    paths = match(node_type_seq, edge_type_seq, graph, node_type_attr,
                  edge_type_attr)
    if proj_type != "graph":
        graph = build_subgraph(paths, graph)
    return {
        "graph": graph,
        "paths": paths,
        "node_alias": node_alias,
        "edge_alias": edge_alias,
        "node_type_attr": node_type_attr,
        "edge_type_attr": edge_type_attr
    }


def _lookup_attrs(node_alias, graph, to_set, path):
    """
    Helper to get attrs based on set input.

    :param node_alias: Dict.
    :param graph: networkx.Graph
    :param to_set: List of dictionaries.
    :param path: List.
    :returns: Dict.
    """
    attrs = {}
    for i, attr in enumerate(to_set):
        key = attr.get("key", i)
        value = attr.get("value", "")
        if not value:
            lookup = attr.get("value_lookup", "")
            if lookup:
                alias, lookup_key = lookup.split(".")
                alias_index = node_alias[alias]
                node = path[alias_index]
                value = graph.node[node][lookup_key]
        attrs[key] = value
    return attrs


def _get_source_target(node_alias, pattern):
    """
    Uses Node alias system to perform a pattern match.

    :param node_alias: Dict.
    :param pattern: List.
    :returns: Int. Source and target list indices.
    """
    try:
        alias_seq = [p["node"]["alias"] for p in pattern[0::2]]
    except KeyError:
        raise Exception("Please define valid transformation pattern.")
    source = node_alias[alias_seq[0]]
    target = node_alias[alias_seq[-1]]
    return source, target


class NXProjector(object):
    def __init__(self, id_counter):
        """
        This class holds the info and methods necessary for performing the ETL
        actions on a networkx.Graph. It is not a wrapper, and does not store
        the actual graph, just operates on it. Implements match, _project,
        _transfer, and _combine.

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
            return project(source, target, graph, attrs, node_type_attr,
                           edge_type_attr, **kwargs)

        @self.transformation_wrapper("transfer")
        def execute_transfer(source, target, graph, attrs, node_type_attr,
                             edge_type_attr, **kwargs):
            return transfer(source, target, graph, attrs, node_type_attr,
                            edge_type_attr, **kwargs)

        @self.transformation_wrapper("combine")
        def execute_combine(source, target, graph, attrs, node_type_attr,
                            edge_type_attr, node_id="", **kwargs):
            self._id_counter += 1
            p = partial(combine, node_id=int(self._id_counter))
            return p(source, target, graph, attrs, node_type_attr,
                     edge_type_attr, **kwargs)


def reset_index(graph):
    """
    This projection clobbers your nodes, best protect them.

    :param graph: networx.Graph
    :returns: networkx.Graph
    """
    mapping = dict(zip(graph.nodes(), range(0, graph.number_of_nodes())))
    # Change nodes to integers.
    graph = nx.relabel_nodes(graph, mapping)
    return graph


def match(node_type_seq, edge_type_seq, graph, node_type_attr="type",
          edge_type_attr="type"):
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
                             node_type_attr, edge_type_attr)
            path_list.append(paths)
    paths = list(chain.from_iterable(path_list))
    return paths


def project(source, target, graph, attrs={}, node_type_attr="type",
            edge_type_attr="type", **kwargs):
    """
    Executes graph "PROJECT" projection.

    :param source: Int. Source node for transformation.
    :param target: Int. Target node for transformation.
    :param attrs: Dict. Attrs to be set during transformation.
    :param graph: networkx.Graph. Graph of subgraph to transform.
    :returns: networkx.Graph. A projected copy of the wrapped graph
    or its subgraph.
    """
    algorithm = "none"
    over = []
    method = kwargs.get("method", "")
    if method:
        try:
            algorithm = method.keys()[0]
            over = method[algorithm].get("over", [])
        except IndexError:
            raise Exception("Define edge weight calculation method.")
    if algorithm == "jaccard":
        snbrs = {node for node in graph[source].keys()
                 if graph.node[node][node_type_attr] in over}
        tnbrs = {node for node in graph[target].keys()
                 if graph.node[node][node_type_attr] in over}
        intersect = snbrs & tnbrs
        union = snbrs | tnbrs
        jaccard = float(len(intersect)) / len(union)
        attrs["weight"] = jaccard
    if graph.has_edge(source, target):
        edge_attrs = graph[source][target]
        merged_attrs = _merge_attrs(attrs, edge_attrs,
                                    [edge_type_attr, "weight"])
        graph.adj[source][target] = merged_attrs
        graph.adj[target][source] = merged_attrs
    else:
        graph.add_edge(source, target, attrs)
    return graph


def transfer(source, target, graph, attrs={}, node_type_attr="type",
             edge_type_attr="type", **kwargs):
    """
    Execute a graph "TRANSFER" projection.

    :param source: Int. Source node for transformation.
    :param target: Int. Target node for transformation.
    :param attrs: Dict. Attrs to be set during transformation.
    :param graph: networkx.Graph. Graph of subgraph to transform.
    :returns: networkx.Graph. A projected copy of the wrapped graph
    or its subgraph.
    """
    edges = []
    algorithm = "none"
    method = kwargs.get("method", "")
    if method:
        try:
            algorithm = method.keys()[0]
        except IndexError:
            raise Exception("Please define a valid method.")
    if algorithm == "edges" or algorithm == "none":
        nbrs = graph[source]
        edges = zip([target] * len(nbrs), nbrs,
                    [v for (k, v) in nbrs.items()])
    if algorithm == "attrs" or algorithm == "none":
        old_attrs = graph.node[target]
        merged_attrs = _merge_attrs(attrs, old_attrs,
                                    [node_type_attr])
        graph.node[target] = merged_attrs
    graph = _add_edges_from(graph, edges)
    return graph


def combine(source, target, graph, attrs={}, node_type_attr="type",
            edge_type_attr="type", node_id="", **kwargs):

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
    # Filter out newly created nodes from neighbors.
    nbrs = {k: v for (k, v) in nbrs.items()
            if graph.node[k][node_type_attr] != node_type}
    edges = zip([node_id] * len(nbrs), nbrs,
                [v for (k, v) in nbrs.items()])
    graph = _add_edges_from(graph, edges)
    return graph


def traverse(start, node_type_seq, edge_type_seq, graph,
             node_type_attr="type", edge_type_attr="type"):
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
            paths.append(list(stack))
            # Backtrack and keep checking.
            stack.pop()
            current = stack[-1]
            depth -= 1
    return paths


def build_subgraph(paths, graph):
    """
    Takes the paths returned by match and builds a graph.
    :param paths: List of lists.
    :returns: networkx.Graph. Matched sugraph.
    """
    g = nx.Graph()
    for path in paths:
        combined_paths = _combine_paths(path)
        for edges in combined_paths:
            attrs = graph[edges[0]][edges[1]]
            g.add_edge(edges[0], edges[1], attrs)
    for node in g.nodes():
        g.node[node] = dict(graph.node[node])
    return g


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
            merged_attrs = _merge_attrs(attrs, edge_attrs,
                                        [edge_type_attr, "weight"])
            graph.adj[source][target] = merged_attrs
            graph.adj[target][source] = merged_attrs
        else:
            graph.add_edge(source, target, attrs)
    return graph


def _merge_attrs(new_attrs, old_attrs, reserved=[]):
    """
    Merges dicts handling repeated values as mulitvalued attrs using sets.

    :param new_attrs: Dict.
    :param old_attrs: Dict.
    :reserved: List. A list of attributes that cannot have more than value.
    :returns: Dict.
    """
    attrs = {}
    attrs.update(dict(old_attrs))
    for k, v in new_attrs.items():
        if k in reserved:
            attrs[k] = v
        elif k not in attrs:
            attrs[k] = set([v])
        else:
            val = attrs[k]
            if not isinstance(val, set):
                attrs[k] = set([val])
            attrs[k].update([v])
    return attrs


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
