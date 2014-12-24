# -*- coding: utf-8 -*-
from itertools import chain
import networkx as nx
from grammar import parser


def error_handler(fn):
    """
    Wraps the execute method. Will do error handling here.
    """
    def wrapper(self, query, **kwargs):
        try:
            graph = fn(self, query)
        except:
            raise Exception("Check query and graph. An error occurred.")
        return graph
    return wrapper


class Projection(object):
    def __init__(self, graph, node_type_attr="type", edge_type_attr="type"):
        """
        Main class for generating graph projections and schema modifications.
        Wraps a NetworkX graph, and then executes a query written in the
        ProjX query language over the graph. ProjX queries are based on
        Neo4j"s Cypher syntax, but are currently considerably simpler;
        However, they do add a few verbs to Cypher"s vocabulary, namely:
        "TRANSFER" and "PROJECT".

        :param graph: An multi-partite (multi-type) instance of
            networkx.Graph().

        :param node_type_attr: A string node attribute name that distinguishes
            between types (modes). Default is "type".
        :param node_type_attr: A string node attribute name that distinguishes
            between types (modes). Default is "type".

        """
        for node in graph.nodes():
            graph.node[node]["visited_from"] = []
        self.graph = graph
        self.node_type = node_type_attr
        self.edge_type = edge_type_attr
        self.parser = parser
        self._operations = {}
        self._operations_init()

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
        def execute_project(graph, paths, mp, pattern, obj, pred_clause):
            return self._project(graph, paths, mp, pattern, obj, pred_clause)

        @self.operations_wrapper("transfer")
        def execute_transfer(graph, paths, mp, pattern, obj, pred_clause):
            return self._transfer(graph, paths, mp, pattern, obj, pred_clause)

    def _clear(self, nbunch):
        """
        Used to clear the visited attribute on a bunch of nodes.

        :param nbunch: Iterable. A bunch of nodes.
        """
        for node in nbunch:
            self.graph.node[node]["visited_from"] = []

    #@error_handler
    def execute(self, query):
        '''
        This takes a ProjX query and executes it.

        :param query: String. A ProjX query.
        :returns: networkx.Graph. The graph or subgraph with the required
                  schema modfications.
        ''' 
        clauses = self.parser.parseString(query)
        match = clauses[0]
        obj = match.get("object", "")
        pattern = match["pattern"]
        pred_clause = match.get("predicates", "")
        mp = _MatchPattern(pattern)  # Fix pattern processor    
        graph, paths = self._match(mp, obj=obj, pred_clause=pred_clause)
        for clause in clauses[1:]:
            verb = clause["verb"]
            # Here I can check for match to create second subgraph.
            obj = clause.get("object", "")
            pattern = clause["pattern"]
            pred_clause = clause.get("predicates", "")
            operation = self.operations[verb]    
            graph, paths = operation(graph, paths, mp, pattern,
                                     obj, pred_clause)
        return graph

    def match(self, paths):
        """
        Will form part of programmatic api.

        :param path: List of lists. The paths matched
                     by the _match method.
        :returns: networkx.Graph. A matched subgraph.
        """
        pass

    def _match(self, mp, graph=None, obj=None, pred_clause=None):
        """
        Executes traversals to perform initial match on pattern.

        :param pattern: String. A valid pattern string.
        :returns: List of lists. The matched paths.
        """

        if not graph:
            graph = self.graph.copy()
        node_type_seq = mp.node_type_seq
        edge_type_seq = mp.edge_type_seq
        start_type = node_type_seq[0]
        path_list = []
        for node, attrs in graph.nodes(data=True):
            if attrs[self.node_type] == start_type or not start_type:
                paths = self.traverse(node, node_type_seq[1:], edge_type_seq)
                path_list.append(paths)
        paths = list(chain.from_iterable(path_list))
        if not paths:
            raise Exception("There are no nodes matching "
                            "the given type sequence. Check for "
                            "input errors.")
        if obj != 'graph':
            graph = self.build_subgraph(paths)
        return graph, paths

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

    def _project(self, graph, paths, mp, 
                 pattern, obj=None, pred_clause=None):
        """
        Executes graph "PROJECT" projection.

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
        removals = set()
        to_set = ''
        method = ''
        if pred_clause:
            to_set, delete, method = _process_predicate(pred_clause, mp)   
        source, target = _get_source_target(paths, mp, pattern)
        new_edges = []
        for path in paths:
            source_node = path[source]
            target_node = path[target]
            if source < target:
                remove = path[source + 1:target]
            else:
                remove = path[target + 1:source]
            new_attrs = {}
            if to_set:
                new_attrs = _transfer_attrs(new_attrs, to_set, mp,
                                            path, graph, self.node_type)
            if ((method == "jaccard" or not method) and 
                abs(source - target) == 2):
                # Calculate Jaccard index for edge weight.
                snbrs = set(graph[source_node])
                tnbrs = set(graph[target_node])
                intersect = snbrs & tnbrs
                union = snbrs | tnbrs
                jaccard = float(len(intersect)) / len(union) 
                new_attrs["weight"] = jaccard
            removals.update(remove)
            new_edges.append((source_node, target_node, new_attrs))
        graph = self.add_edges_from(graph, new_edges)
        graph.remove_nodes_from(removals)
        return graph, paths

    def transfer(self, mp):
        """
        Performs match, executes _transfer, and returns graph. This can be
        part of programmatic API.

        :param mp: _MatchPattern. The initital pattern specified
                              in "MATCH" statement or in one-line query.
        :returns: networkx.Graph. A projected copy of the wrapped graph
                  or its subgraph.
        """
        pass

    def _transfer(self, graph, paths, mp, 
                  pattern, obj=None, pred_clause=None):
        """
        Execute a graph "TRANSFER" projection.

        :param graph: networkx.Graph. A copy of the wrapped grap or its
                      subgraph.
        :param path: List of lists. The paths matched
                     by the _match method based.
        :param mp: _MatchPattern. The initital pattern specified
                              in "MATCH" statement or in one-line query.
        :param pattern: Optional. String. A valid pattern string. Needed for
                        multi-line query.
        :param edges: Bool. Default False. Settings this to true executes a
                      merge.
        :returns: networkx.Graph. A projected copy of the wrapped graph
                  or its subgraph.
        """
        removals = set()
        delete = []
        to_set = ''
        if pred_clause:
            to_set, delete, method = _process_predicate(pred_clause, mp)
        source, target = _get_source_target(paths, mp, pattern)
        for path in paths:
            transfer_source = path[source]
            transfer_target = path[target]
            if delete:
                for i in delete:
                    removals.update([path[i]])
            if obj == "edges" or not obj:
                edges = graph[transfer_source]
                # This is naive.
                new_edges = zip(
                    [transfer_target] * len(edges),
                    edges,
                    [v for (k, v) in edges.items()]
                )
                graph = self.add_edges_from(graph, new_edges)
            if (obj == "attrs" or not obj) and to_set:
                attrs = graph.node[transfer_target]
                new_attrs = _transfer_attrs(attrs, to_set, mp, path, 
                                            graph, self.node_type)
                graph.node[transfer_target] = new_attrs
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
                        self.edge_type,
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
                            (attrs[self.node_type] == node_type_seq[depth] or
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

    def add_edges_from(self, graph, new_edges):
        """
        An alternative to the networkx.Graph.add_edges_from.
        Handles non-reserved attributes as sets.
        """
        reserved = ['weight', self.edge_type]
        for source, target, attrs in new_edges:
            if graph.has_edge(source, target):
                edge_attrs = graph[source][target]
                for k, v in attrs.items():
                    if k in reserved:
                        edge_attrs[k] = v
                    elif k not in edge_attrs:
                        edge_attrs[k] = set([v])
                    else:
                        val = edge_attrs[k]
                        if not isinstance(val, set):
                            edge_attrs[k] = set([val])
                        edge_attrs[k].update(v)
            else:
                graph.add_edge(source, target, attrs)
        return graph

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


def _transfer_attrs(attrs, to_set, mp, path, graph, node_type):
    """
    Add new attributes to old attributes during a transfer or 
    project. Handles non-reserved attributes
    using sets.
    """
    new_attrs = dict(attrs)
    for att in to_set:
        attr1 = att['attr1']
        attr2 = att['attr2']
        type2 = att.get('type2', '')
        # Get the attribute value a to be set.
        if type2:
            i = mp.node_alias[type2]
            node = path[i]
            a = graph.node[node][attr2]
        else:
            a = attr2
        # Unless reserved, use sets to hold edge attrs.
        if attr1 == node_type:
            new_attrs[node_type] = a 
        elif attr1 in new_attrs:
            val = new_attrs[attr1]
            if not isinstance(val, set):
                new_attrs[attrs1] = set([val])
            new_attrs[attr1].update([a])
        else:
            new_attrs[attr1] = set([a])  
    return new_attrs


def _process_predicate(pred_clause, mp):
    """
    Iterate over predicate clauses and prepare necessary input
    for operations.

    :param pred_clause: Container of predicate clauses generated by parser.
    :param mp: _MatchPattern. Object containing alias mappings.

    :return to_set: List. A list of dict containing attr values.
    :return delete: List. A list of int nodes to delete.
    :method: String. A method for calculating edge weight during projection.

    """
    to_set = []
    delete = []
    method = ''   
    preds = pred_clause[0]['pred_clauses']
    for pred in preds:
        p = pred['predicate']
        if p == 'set':
            to_set = pred['pred_objects']
        elif p == 'delete':
            delete = [
                mp.node_alias[p] for p in pred['pred_objects']
            ]
        elif p == 'method':
            method = pred['pred_objects'][0]
    return to_set, delete, method


def _get_source_target(paths, mp, pattern):
    """
    Uses _MatchPattern"s alias system to perform a pattern match.

    :param mp: _MatchPattern. The initital pattern specified
                          in "MATCH" statement or in one-line query.
    :param pattern: String. A valid pattern string of aliases.
    """
    alias_seq = [p[0] for p in pattern]
    source = mp.node_alias[alias_seq[0]]
    target = mp.node_alias[alias_seq[-1]]
    return source, target


class _MatchPattern(object):

    def __init__(self, pattern):
        """
        This is a helper class that takes a match pattern and
        maintains an alias dictionary. This allows for multi-line
        queries to utilize aliases.

        :param pattern: String. A ProjX language pattern.
        """
        self.pattern = pattern
        self.nodes = pattern["nodes"]
        self.edges = pattern["edges"]
        self.node_alias = {}
        self.edge_alias = {}
        self.node_type_seq = []
        self.edge_type_seq = []
        for i, node in enumerate(self.nodes):
            node = node[0]
            node_alias = node["alias"]
            self.node_alias[node_alias] = i
            tp = node.get("type", "")
            if tp:
                tp = tp[0]
            self.node_type_seq.append(tp)
        for j, edge in enumerate(self.edges):
            if edge:
                edge = edge[0]
                edge_alias = edge["alias"]
                self.edge_alias[edge_alias] = j
                tp = edge.get("type", "")
                if tp:
                    tp = tp[0]
            else:
                tp = ""
                self.edge_alias[""] = j
            self.edge_type_seq.append(tp)
