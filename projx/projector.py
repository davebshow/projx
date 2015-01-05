# -*- coding: utf-8 -*-
from itertools import chain
import networkx as nx


class BaseProjector(object):

    def match(self):
        """
        This method should return a list of paths, or a generator,
        or a stream containing paths. Something that can be passed
        to the transformer/loader.
        """
        raise NotImplementedError()


class NXProjector(BaseProjector):
    def __init__(self, proj_type, query, node_alias, edge_alias,
                 node_type_attr="type", edge_type_attr="type"):
        """
        This class holds the info and methods necessary for performing the ETL
        actions on a networkx.Graph. It is not a wrapper, and does not store
        the actual graph, just operates on it. Implements match, _project,
        _transfer, and _combine.

        :param proj_type:  Str.
        :param query: List/Tuple of two lists. First list contains the node
        type sequence. Second list contains edge type sequence.
        :param node_alias: Dict.
        :param edge_alias: Dict.
        :param node_type_attr: Str. Node attribute name that distinguishes
        between types (modes). Default is "type".
        :param edge_type_attr: Str. Edge attribute name that distinguishes
        between types. Default is "type".

        """
        super(NXProjector, self).__init__()
        self.proj_type = proj_type
        self.query = query
        self.node_alias = node_alias
        self.edge_alias = edge_alias
        self.node_type_attr = node_type_attr
        self.edge_type_attr = edge_type_attr
        self._transformation = {}
        self._transformation_init()

    def process_graph(self, graph):
        """
        This projection clobbers your nodes, best protect them.

        :param graph: networx.Graph
        :returns: networkx.Graph
        """
        mapping = dict(zip(graph.nodes(), range(0, graph.number_of_nodes())))
        # Change nodes to integers.
        graph = nx.relabel_nodes(graph, mapping)
        self.id_counter = max(graph.nodes()) + 1
        return graph

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
        def execute_project(source, target, attrs, graph, **kwargs):
            """
            :param source: Int. Source node for transformation.
            :param target: Int. Target node for transformation.
            :param attrs: Dict. Attrs to be set during transformation.
            :param graph: networkx.Graph. Graph of subgraph to transform.
            :returns: function.
            """
            return self._project(source, target, attrs, graph, **kwargs)

        @self.transformation_wrapper("transfer")
        def execute_transfer(source, target, attrs, graph, **kwargs):
            """
            :param source: Int. Source node for transformation.
            :param target: Int. Target node for transformation.
            :param attrs: Dict. Attrs to be set during transformation.
            :param graph: networkx.Graph. Graph of subgraph to transform.
            :returns: function.
            """
            return self._transfer(source, target, attrs, graph, **kwargs)

        @self.transformation_wrapper("combine")
        def execute_combine(source, target, attrs, graph, **kwargs):
            """
            :param source: Int. Source node for transformation.
            :param target: Int. Target node for transformation.
            :param attrs: Dict. Attrs to be set during transformation.
            :param graph: networkx.Graph. Graph of subgraph to transform.
            :returns: function.
            """
            return self._combine(source, target, attrs, graph, **kwargs)

    def match(self, graph):
        """
        Executes traversals to perform initial match on pattern.

        :param graph: networkx.Graph
        :returns: List of lists. The matched paths.
        """
        node_type_seq, edge_type_seq = self.query
        start_type = node_type_seq[0]
        path_list = []
        for node, attrs in graph.nodes(data=True):
            if attrs[self.node_type_attr] == start_type or not start_type:
                paths = self.traverse(node, node_type_seq[1:],
                                      edge_type_seq, graph)
                path_list.append(paths)
        paths = list(chain.from_iterable(path_list))
        return paths

    def _project(self, source, target, attrs, graph, **kwargs):
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
                     if graph.node[node][self.node_type_attr] in over}
            tnbrs = {node for node in graph[target].keys()
                     if graph.node[node][self.node_type_attr] in over}
            intersect = snbrs & tnbrs
            union = snbrs | tnbrs
            jaccard = float(len(intersect)) / len(union)
            attrs["weight"] = jaccard
        if graph.has_edge(source, target):
            edge_attrs = graph[source][target]
            merged_attrs = _merge_attrs(attrs, edge_attrs,
                                        [self.edge_type_attr, "weight"])
            graph.adj[source][target] = merged_attrs
            graph.adj[target][source] = merged_attrs
        else:
            graph.add_edge(source, target, attrs)
        return graph

    def _transfer(self, source, target, attrs, graph, **kwargs):
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
                                        [self.node_type_attr])
            graph.node[target] = merged_attrs
        graph = self.add_edges_from(graph, edges)
        return graph

    def _combine(self, source, target, attrs, graph, **kwargs):

        """
        Executes graph "COMBINE" projection.

        :param source: Int. Source node for transformation.
        :param target: Int. Target node for transformation.
        :param attrs: Dict. Attrs to be set during transformation.
        :param graph: networkx.Graph. Graph of subgraph to transform.
        :returns: networkx.Graph. A projected copy of the wrapped graph
        or its subgraph.
        """
        node_type = attrs.get(self.node_type_attr, "")
        if not node_type:
            node_type = "{0}_{1}".format(
                graph.node[source][self.node_type_attr],
                graph.node[target][self.node_type_attr]
            )
            attrs[self.node_type_attr] = node_type
        new_node = int(self.id_counter)
        graph.add_node(new_node, attrs)
        self.id_counter += 1
        nbrs = dict(graph[source])
        nbrs.update(dict(graph[target]))
        # Filter out newly created nodes from neighbors.
        nbrs = {k: v for (k, v) in nbrs.items()
                if graph.node[k][self.node_type_attr] != node_type}
        edges = zip([new_node] * len(nbrs), nbrs,
                    [v for (k, v) in nbrs.items()])
        graph = self.add_edges_from(graph, edges)
        return graph

    def traverse(self, start, node_type_seq, edge_type_seq, graph):
        """
        This is a controlled depth, depth first traversal of a NetworkX
        graph and the core of this library. Criteria for searching depends
        on a start node and a sequence of types as designated in the query
        pattern. From the start node, the traversal will visit nodes that
        match the type sequence. It does not allow cycles or backtracking
        along the same path. Could be very memory inefficient in very dense
        graph with 3 + type queries.

        :param start: Integer. Starting point for the traversal.
        :param node_type_seq: List of strings. Derived from the match pattern.
        :param node_type_seq: List of strings. Derived from the match pattern.
        :param graph: networkx.Graph
        :returns: List of lists. All matched paths.
        """
        # Initialize a stack to keep
        # track of traversal progress.
        stack = [start]
        # Store all valid paths based on
        # type sequence.
        paths = []
        # Keep track of visited nodes, later
        # the visited list will be cleared.
        visited = set()
        visited_from = {}
        # The traversal will begin
        # at the designated start point.
        current = start
        # Keep track depth from start node
        # to watch for successful sequence match.
        depth = 0
        # This is the len of a successful sequence.
        max_depth = len(node_type_seq)
        # When the stack runs out, all candidate
        # nodes have been visited.
        while len(stack) > 0:
            # Traverse!
            if depth < max_depth:
                nbrs = set(graph[current]) - set([current])
                for nbr in nbrs:
                    edge_type_attr = graph[current][nbr].get(
                        self.edge_type_attr,
                        None
                    )
                    attrs = graph.node[nbr]
                    # Here check candidate node validity.
                    # Make sure this path hasn"t been checked already.
                    # Make sure it matches the type sequence.
                    # Make sure it"s not backtracking on same path.
                    # Kind of a nasty if, but I don"t want to
                    # make a method call.
                    visited_from.setdefault(nbr, [])
                    if (current not in visited_from[nbr] and
                            nbr not in stack and
                            (edge_type_attr == edge_type_seq[depth] or
                             edge_type_seq[depth] == "") and
                            (attrs[self.node_type_attr] == node_type_seq[depth]
                             or node_type_seq[depth] == "")):
                        visited_from[nbr].append(current)
                        visited.update([nbr])
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
        # Clear the visited attribute to prepare
        # for next start node to begin traversal.
        #self._clear(visited, graph)
        return paths

    def build_subgraph(self, paths, graph):
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

    def add_edges_from(self, graph, edges):
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
                                            [self.edge_type_attr, "weight"])
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
