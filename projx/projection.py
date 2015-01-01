# -*- coding: utf-8 -*-
from itertools import chain
import networkx as nx
from grammar import parser


def execute_etl(graph, etl):
    etl = _ETL(etl)
    # Extractor is a function that returns a Projection class.
    extractor = etl.extractor
    # Loader can be a class or function that contains transformer.
    loader = etl.loader
    # Projection is a class that implements the match, _project, _transform,
    # and _combine methods.
    projection = extractor(graph)
    # Return paths or stream to be passed to loader transformer.
    paths = projection.match(etl.query)
    # Loader should be able to work using just the _ETL object, a projection
    # object, and the paths generator.
    graph = loader(etl, projection, paths)
    return graph


class _ETL(object):

    def __init__(self, etl):
        """
        A helper class that parses the ETL JSON, and initializes the
        required extractor, transformers, and loader. Also store key
        variable used in transformation/loading.
        """
        # Get the extractor info.
        try:
            self._extractor = etl["extractor"]
            self.extractor_name = self._extractor.keys()[0]
        except (KeyError, IndexError):
            raise Error("Please define valid extractor")
        # Get the loader info.
        try:
            self._loader = etl["loader"]
            self.loader_name = self._loader.keys()[0]
        except (KeyError, IndexError):
            raise Error("Please define valid loader.")

        # Get the extractor function.
        self._extractors = {}
        self._init_extractors()
        self.extractor = self.extractors[self.extractor_name]

        self.transformers = etl.get("transformers", [])

        # Get the loader function.
        self._loaders = {}
        self._init_loaders()
        self.loader = self.loaders[self.loader_name]

    def _get_extractors(self):
        return self._extractors 
    extractors = property(fget=_get_extractors)

    def extractors_wrapper(self, extractor):
        def wrapper(fn):
            self.extractors[extractor] = fn
        return wrapper

    def _init_extractors(self):
        """
        Update extractors dict to allow for extensible extractor 
        functionality. Here we set the query attribute that will be passed to
        extractor match method. Also, each extractor function will return
        a Projection class object that defines a match method.
        """
        @self.extractors_wrapper("networkx")
        def nx_extractor(graph):
            # Set all of the attrs required for NetworkX work.
            # This is the traversal pattern
            node_type_attr = self._extractor[self.extractor_name].get(
                "node_type_attr", "type"
            )
            edge_type_attr = self._extractor[self.extractor_name].get(
                "edge_type_attr", "type"
            )
            traversal = self._extractor[self.extractor_name].get(
                "traversal", []
            )
            nodes = traversal[0::2]
            edges = traversal[1::2]
            # This determines whether to extract the whole graph or just a subgraph
            self.subgraph = self._extractor[self.extractor_name].get(
                "class", "subgraph"
            )
            try:
                # Give the ETL object some node alias info
                self.node_alias = {node["node"]["alias"]: i for 
                                   (i, node) in enumerate(nodes)}
                self.edge_alias = {edge["edge"].get("alias", i): i for 
                                   (i, edge) in enumerate(edges)}
                node_type_seq = [node["node"].get(node_type_attr, "") 
                                 for node in nodes]
                edge_type_seq = [edge["edge"].get(edge_type_attr, "") 
                                 for edge in edges]
            except KeyError:
                raise Error("Please define valid traversal sequence")
            self.query = (node_type_seq, edge_type_seq)
            return NXProjection(graph, node_type_attr, edge_type_attr)

    def _get_loader(self):
        return self._loaders 
    loaders = property(fget=_get_loader)

    def loaders_wrapper(self, loader):
        def wrapper(fn):
            self.loaders[loader] = fn
        return wrapper

    def _init_loaders(self):
        """
        Update loaders dict to allow for extensible loader functionality.
        Returns a class/function that performs transformation and loads graph.
        """
        @self.loaders_wrapper("networkx")
        def nx_extractor(etl, projection, paths):
            return nx_loader(etl, projection, paths)
    

# Functions to load and transform NetworkX graph
def nx_loader(etl, projection, paths):
    if etl.subgraph == "graph":
        graph = projection.copy()
    else:
        graph = projection.build_subgraph(paths)   
    if len(etl.transformers) > 1: 
        graph = nx_transformer_pipeline(etl, projection, graph, paths)
    elif len(etl.transformers) == 1:
        graph = nx_transformer(etl, projection, graph, paths)
    return graph


def nx_transformer(etl, projection, graph, paths):
    removals = set()
    transformer = etl.transformers[0]
    transformation = transformer.keys()[0]
    pattern = transformer[transformation]["pattern"]
    source, target = _get_source_target(etl, pattern)
    to_set = transformer[transformation].get("set", [])
    fn = projection.operations[transformation]
    delete_alias = transformer[transformation].get("delete", {}).get("alias", [])
    to_delete = [etl.node_alias[alias] for alias in delete_alias]
    method = transformer[transformation].get("method", {})
    for path in paths:
        source_node = path[source]
        target_node = path[target]
        attrs = {}
        for i, attr in enumerate(to_set):
            key = attr.get("key", i)
            value = attr.get("value", "")
            if not value:
                lookup = attr.get("value_lookup", "")
                if lookup:
                    alias, lookup_key = lookup.split(".")
                    alias_index = etl.node_alias[alias]
                    node = path[alias_index]
                    value = graph.node[node][lookup_key]
            attrs[key] = value
        graph = fn(source_node, target_node, attrs, graph, method=method)
        for i in to_delete:
            removals.update([path[i]])
    graph.remove_nodes_from(removals)
    return graph


def nx_transformer_pipeline(etl, projection, graph, paths):
    """
    Map transformation rules on paths, and then apply to graph.
    :returns: Graph. 
    """
    removals = set()
    for path in paths:
        for transformer in etl.transformers:
            transformation = transformer.keys()[0]
            pattern = transformer[transformation]["pattern"]
            source, target = _get_source_target(etl, pattern)
            source_node = path[source]
            target_node = path[target]
            to_set = transformer[transformation].get("set", [])
            method = transformer[transformation].get("method", {})
            attrs = {}
            for i, attr in enumerate(to_set):
                key = attr.get("key", i)
                value = attr.get("value", "")
                if not value:
                    lookup = attr.get("value_lookup", "")
                    if lookup:
                        alias, lookup_key = lookup.split(".")
                        alias_index = etl.node_alias[alias]
                        node = path[alias_index]
                        value = graph.node[node][lookup_key]
                attrs[key] = value
            fn = projection.operations[transformation]
            graph = fn(source_node, target_node, attrs, graph, method=method)
            delete_alias = transformer[transformation].get("delete", {}).get("alias", [])
            to_delete = [etl.node_alias[alias] for alias in delete_alias]
        for i in to_delete:
            removals.update([path[i]])
    graph.remove_nodes_from(removals)
    return graph


def _get_source_target(etl, pattern):
    """
    Uses _ETL's alias system to perform a pattern match.

    :param etl: _ETL. The initital pattern specified
                          in "MATCH" statement or in one-line query.
    :param pattern: List.
    :returns: Int. Source and target list indices.
    """
    try:
        alias_seq = [p["node"]["alias"] for p in pattern[0::2]]
    except KeyError:
        raise Error("Please define valid transformation pattern.")
    source = etl.node_alias[alias_seq[0]]
    target = etl.node_alias[alias_seq[-1]]
    return source, target


class BaseProjection(object):

    def match(self):
        """
        This method should return a list of paths, or a generator,
        or a stream containing paths. Something that can be passed 
        to the transformer/loader.
        """
        raise NotImplementedError()


class NXProjection(BaseProjection):
    def __init__(self, graph, node_type_attr="type", edge_type_attr="type"):
        """
        Main class for generating graph projections and schema modifications.
        Wraps a NetworkX graph, and then executes a query written in the
        ProjX query language over the graph. ProjX queries are based on
        Neo4j"s Cypher syntax, but are currently considerably simpler.
        "TRANSFER" and "PROJECT" and "COMBINE".

        :param graph: An multi-partite (multi-type) instance of
            networkx.Graph().
        :param node_type_attr: A string node attribute name that distinguishes
            between types (modes). Default is "type".
        :param node_type_attr: A string node attribute name that distinguishes
            between types (modes). Default is "type".

        """
        # Preserve the original node in an attribute called node,
        # then relabel the nodes with integers.
        super(NXProjection, self).__init__()
        for node in graph.nodes():
            graph.node[node]["visited_from"] = []
            graph.node[node]["node"] = node
        mapping = dict(zip(graph.nodes(), range(0, graph.number_of_nodes())))
        self.graph = nx.relabel_nodes(graph, mapping)
        self.node_type_attr = node_type_attr
        self.edge_type_attr = edge_type_attr
        self.parser = parser
        self._operations = {}
        self._operations_init()

    def copy(self):
        return self.graph.copy()

    def operations_wrapper(self, verb):
        """
        Wraps the operations methods and adds them to the operationss dictionary for
        easy retrivial during query parse.

        :param verb: String. The ProjX verb assiociated with the wrapped
            function.
        """
        def wrapper(fn):
            self._operations[verb] = fn
        return wrapper

    def _get_operations(self):
        """
        Return operations for operationss property.

        :returns: Dict. A dict containing a mapping of verbs to operations
            methods.
        """
        return self._operations
    operations = property(fget=_get_operations)

    def _operations_init(self):
        """
        A series of functions representing the grammar operationss. These are
        wrapped by the operationss wrapper and added to the operationss dict.
        Later during the parsing and execution phase these are called as
        pointers to the various graph transformation methods
        (transfer and project).
        """
        @self.operations_wrapper("project")
        def execute_project(source, target, attrs, graph, **kwargs):
            return self._project(source, target, attrs, graph, **kwargs)

        @self.operations_wrapper("transfer")
        def execute_transfer(source, target, attrs, graph, **kwargs):
            return self._transfer(source, target, attrs, graph, **kwargs)

        @self.operations_wrapper("combine")
        def execute_combine(source, target, attrs, graph, **kwargs):
            return self._combine(source, target, attrs, graph, **kwargs)

    def _clear(self, nbunch):
        """
        Used to clear the visited attribute on a bunch of nodes.

        :param nbunch: Iterable. A bunch of nodes.
        """
        for node in nbunch:
            self.graph.node[node]["visited_from"] = []

    def match(self, query):
        """
        Executes traversals to perform initial match on pattern.

        :param pattern: String. A valid pattern string.
        :returns: List of lists. The matched paths.
        """
        node_type_seq, edge_type_seq = query
        start_type = node_type_seq[0]
        path_list = []
        for node, attrs in self.graph.nodes(data=True):
            if attrs[self.node_type_attr] == start_type or not start_type:
                paths = self.traverse(node, node_type_seq[1:], edge_type_seq)
                path_list.append(paths)
        paths = list(chain.from_iterable(path_list))
        return paths

    def project(self, mp):
        """
        Performs match, executes _project, and returns graph. This can be
        part of programmatic API.

        :param mp: _MatchPattern. The initital pattern specified
                              in "MATCH" statement or in one-line query.
        :returns: networkx.Graph. A projected copy of the wrapped graph
                  or its subgraph.

        """
        pass

    def _project(self, source, target, attrs, graph, **kwargs):
        """
        Executes graph "PROJECT" projection.

        :returns: networkx.Graph. A projected copy of the wrapped graph
                  or its subgraph.
        """


        # Ugly
        method = kwargs.get("method", "")
        if method:
            try:
                algo = method.keys()[0]
                over = method[algo].get("over", [])
            except IndexError:
                raise Error("Please define edge weight calculation method.")
        else:
            algo = ""
            over = []
        if algo == "jaccard" or not algo:
            snbrs = {node for node  in graph[source].keys()
                     if graph.node[node][self.node_type_attr] in over}
            tnbrs = {node for node in graph[target].keys()
                     if graph.node[node][self.node_type_attr] in over}
            intersect = snbrs & tnbrs
            union = snbrs | tnbrs
            jaccard = float(len(intersect)) / len(union)
            attrs["weight"] = jaccard
        if graph.has_edge(source, target):
            edge_attrs = graph[source][target]
            merged_attrs = self._merge_attrs(attrs, edge_attrs, 
                                             self.node_type_attr)
            graph.adj[source][target] = merged_attrs
            graph.adj[target][source] = merged_attrs
        else:
            graph.add_edge(source, target, attrs)
        return graph

    def _transfer(self, source, target, attrs, graph, **kwargs):
        """
        Execute a graph "TRANSFER" projection.

        :returns: networkx.Graph. A projected copy of the wrapped graph
                  or its subgraph.
        """
        method = kwargs.get("method", "")
        if method:
            try:
                algo = method.keys()[0]
            except IndexError:
                raise Error("Please define a valid method.")  
        else:
            algo = ''         
        if algo == "edges" or not algo:
            nbrs = graph[source]
            new_edges = zip(
                [target] * len(nbrs),
                nbrs,
                [v for (k, v) in nbrs.items()]
            )
        if algo == "attrs" or not algo:
            old_attrs = graph.node[target]
            merged_attrs = self._merge_attrs(attrs, old_attrs,
                                             self.node_type_attr)
            graph.node[target] = merged_attrs
        graph = self.add_edges_from(graph, new_edges)
        return graph

    def _combine(self, graph, paths, mp, pattern, obj=None, pred_clause=None):

        """
        Executes graph "COMBINE" projection. Ooofs.

        :param graph: networkx.Graph. A copy of the wrapped grap or its
                      subgraph.
        :param path: List of lists. The paths matched
                     by the _match method based.
        :param mp: _MatchPattern. The initital pattern specified
                              in "MATCH" statement or in one-line query.
        :param pattern: Optional. String. A valid pattern string. Needed for
                        multi-line query.
        :returns: networkx.Graph. A projected copy of the wrapped graph
                  or its subgraph.
        """
        id_counter = max(graph.nodes()) + 1
        removals = set()
        delete = []
        to_set = ''
        node_ids = {}
        new_edges = []
        if pred_clause:
            to_set, delete, method = _process_predicate(pred_clause, mp)
        source, target = _get_source_target(paths, mp, pattern)
        null_types = [mp.node_type_seq[i] for i in delete]
        for path in paths:
            source_node = path[source]
            target_node = path[target]
            node_id = node_ids.get(target_node, "")        
            for i in delete:
                removals.update([path[i]])
            # Set attrs.
            if node_id:
                attrs = graph.node[node_id]
            else:
                attrs = {}
            if to_set:
                new_attrs = _transfer_attrs(attrs, to_set, mp,
                                            path, graph, self.node_type)
            # Set up a combo type if not specified.
            if not new_attrs.get(self.node_type, ""):
                new_attrs[self.node_type] = '{0}_{1}'.format(
                    graph.node[source_node][self.node_type],
                    graph.node[target_node][self.node_type]
                )
            # Create nodes.
            if node_id:
                graph.node[node_id] = new_attrs
            else:
                node_ids[target_node] = id_counter
                new_node = id_counter
                graph.add_node(new_node, new_attrs)
                id_counter += 1
            # Build edges.
            nbrs = dict(graph[source_node]) # Copy
            nbrs.update(dict(graph[target_node]))
            # Have to check speed here with a bigger graph.
            nbrs = {k: v for (k, v) in nbrs.items() 
                    if graph.node[k][self.node_type] not in null_types}
            new_edges += zip(
                [new_node] * len(nbrs),
                nbrs,
                [v for (k, v) in nbrs.items()]
            )
        graph = self.add_edges_from(graph, new_edges)
        graph.remove_nodes_from(removals)
        return graph, paths

    def traverse(self, start, node_type_seq, edge_type_seq):
        """
        This is a controlled depth, depth first traversal of a NetworkX
        graph and the core of this library. Criteria for searching depends
        on a start node and a sequence of types as designated in the query
        pattern. From the start node, the traversal will visit nodes that
        match the type sequence. It does not allow cycles or backtracking
        along the same path. Could be very memory inefficient in very dense
        graph with 3 + type queries.

        :param start: Integer. Starting point for the traversal.
        :param type_seq: List of strings. Derived from the match pattern.
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
                nbrs = set(self.graph[current]) - set([current])
                for nbr in nbrs:
                    edge_type_attr = self.graph[current][nbr].get(
                        self.edge_type_attr,
                        None
                    )
                    attrs = self.graph.node[nbr]
                    # Here check candidate node validity.
                    # Make sure this path hasn"t been checked already.
                    # Make sure it matches the type sequence.
                    # Make sure it"s not backtracking on same path.
                    # Kind of a nasty if, but I don"t want to
                    # make a method call.
                    if (current not in attrs["visited_from"] and
                            nbr not in stack and
                            (edge_type_attr == edge_type_seq[depth] or
                             edge_type_seq[depth] == "") and
                            (attrs[self.node_type_attr] == node_type_seq[depth] or
                             node_type_seq[depth] == "")):
                        self.graph.node[nbr]["visited_from"].append(current)
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
        self._clear(visited)
        return paths

    def build_subgraph(self, paths):
        """
        Takes the paths returned by _match and builds a graph.
        :param paths: List of lists. The paths matched
                      by the _match method based.
        :returns: networkx.Graph. A subgraph of the wrapped graph.
        """
        g = nx.Graph()
        for path in paths:
            combined_paths = _combine_paths(path)
            for edges in combined_paths:
                attrs = self.graph[edges[0]][edges[1]]
                g.add_edge(edges[0], edges[1], attrs)
        for node in g.nodes():
            g.node[node] = dict(self.graph.node[node])
        return g

    def add_edges_from(self, graph, new_edges):
        """
        An alternative to the networkx.Graph.add_edges_from.
        Handles non-reserved attributes as sets.
        """
        for source, target, attrs in new_edges:
            if graph.has_edge(source, target):
                edge_attrs = graph[source][target]
                merged_attrs = self._merge_attrs(
                    attrs,
                    edge_attrs,
                    [self.edge_type_attr, "weight"]
                )
                graph.adj[source][target] = merged_attrs
                graph.adj[target][source] = merged_attrs
            else:
                graph.add_edge(source, target, attrs)
        return graph

    def _merge_attrs(self, new_attrs, old_attrs, reserved=[]):
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


