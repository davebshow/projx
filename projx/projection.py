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
            raise Exception('Check query and graph. An error occurred.')
        return graph
    return wrapper


class Projection(object):
    def __init__(self, graph, node_type_attr='type', edge_type_attr='type'):
        """
        Main class for generating graph projections and schema modifications.
        Wraps a NetworkX graph, and then executes a query written in the
        ProjX query language over the graph. ProjX queries are based on
        Neo4j's Cypher syntax, but are currently considerably simpler -
        lacking predicates, multiple matches, return statements, ordering etc.
        However, they do add a few verbs to Cypher's vocabulary, namely:
        "TRANSFER" and "PROJECT". "TRANSFER" allows to transfer the attributes
        of nodes of a designated type to all neighboring nodes of another
        designated type. "TRANSFER" also has a variation "MERGE". "MERGE"
        performs a normal "TRANSFER", but also transfers edges to the transfer
        target There are two types of queries. The first type are Matched
        pattern queries that act upon a pattern matched subgraph
        of the wrapped graph. The second type consists of statements
        that apply schema modifications across the entire wrapped graph.
        Query language specifications are included in the documentation for
        the execute method

        :param graph: An multi-partite (multi-type) instance of
            networkx.Graph().

        :param node_type_attr: A string node attribute name that distinguishes
            between types (modes). Default is 'type'.
        :param node_type_attr: A string node attribute name that distinguishes
            between types (modes). Default is 'type'.

        """
        for node in graph.nodes():
            graph.node[node]['visited_from'] = []
        self.graph = graph
        self.node_type = node_type_attr
        self.edge_type = edge_type_attr
        self._removals = set()
        self.parser = parser
        self._actions = {}
        # Updates the actions with the defined rules.
        self._action_rules()

    def action_wrapper(self, verb):
        """
        Wraps the action methods and adds them to the actions dictionary for
        easy retrivial during query parse.

        :param verb: String. The ProjX verb assiociated with the wrapped
            function.
        """
        def wrapper(fn):
            self._actions[verb] = fn
        return wrapper

    def _get_actions(self):
        """
        Return action for actions property.

        :returns: Dict. A dict containing a mapping of verbs to action
            methods.
        """
        return self._actions
    actions = property(fget=_get_actions)

    def _action_rules(self):
        """
        A series of functions representing the grammar actions. These are
        wrapped by the actions wrapper and added to the actions dict.
        Later during the parsing and execution phase these are called as
        pointers to the various graph transformation methods
        (transfer and project).
        """
        @self.action_wrapper('project')
        def execute_project(graph, paths, mp, pattern, obj):
            return self._project(graph, paths, mp, pattern, obj)

        @self.action_wrapper('transfer')
        def execute_transfer(graph, paths, mp, pattern, obj):
            return self._transfer(graph, paths, mp, pattern, obj)


    def _clear(self, nbunch):
        """
        Used to clear the visited attribute on a bunch of nodes.

        :param nbunch: Iterable. A bunch of nodes.
        """
        for node in nbunch:
            self.graph.node[node]['visited_from'] = []

    #@error_handler
    def execute(self, query):
        """
        This takes a ProjX query and executes it.
        The ProjX syntax is as follows:

        Verbs:
        ------
        - "MATCH" Matches a pattern of nodes based on type.
        - "MATCH PARTIAL" Like "MATCH", but also matchs a partial pattern.
                Coming soon...
        - "TRANSFER" Merges the edges and attributes of nodes of one
                type across a specified sequence of neighboring nodes
                to nodes of another type.
        - "PROJECT" Projects a relationship between nodes of one
                type across a specified sequence of neighboring nodes.
        - "RETURN" Specify table/graph and nodes to return. Coming soon...

        Objects:
        --------
        Objects act as parameters that can be passed to verbs.

        - "ATTRS" When used with "TRANSFER" only attributes will be 
                transfered.
        - "EDGES" When used with "TRANSFER" only edges will be 
                transfered.

        Patterns:
        =========

        Nodes:
        ------
        Nodes are represented using (). For the minimal syntax, the
        () contains at least a node type specification, this specification
        corresponds to the attribute set at init node_type_attr: (Type1).
        For longer queries over subgraphs, it is recommended to
        include an alias with the (): (t1:Type1). This allows for
        cleaner code and prevents errors when using complex pattern
        that repeat types.
            - (f:Foo)
            - (b:Bar)

        Edges:
        ------
        Currently, only simple undirected edges are permited. They are
        denoted with the hyphen -. Support for edge types and attrs are
        coming soon.

        Patterns:
        ---------
        A pattern is a combination of nodes and edges. It creates a
        "type sequence", or a set of criteria that determain a legal
        path during graph traversal based on node's node_type_attr. For
        example, if we want to locate all nodes with node_type_attr == 'Type1'
        that are connected to nodes with node_type_attr == 'Type2', the pattern
        would be specified as "(t1:Type1)-(t2:Type2)". A pattern can be as
        long as necessary, and can repeat elements. Note that the traversal
        does not permit cycles.
            - "(f1:Foo)-(b:Bar)-(f2:Foo)"
            - "(d:Dog)-(p1:Person)-(p2:Person)-(c:Cat)"

        Queries:
        ========
        ProjX queries combine a verb with a pattern to perform some kind
        of search or schema modification over the graph. If queries begin
        with a "MATCH" clause, they will project across only the matched
        subgraph, thus discarding all nodes that do not match. If queries
        begin with a different clause, they will still only be able to act
        upon the first matched pattern; however, they retain all other nodes
        not involved in the pattern regards of other operations.

        Matched subgraph queries:
        -------------------------
        Matched subgraph queries must begin with a "MATCH" statement. This
        produces the subgraph upon which the rest of the verbs will
        operate. After a graph is match, other projections can be perfomed
        upon the resulting subgraph. For example, let's imagine we want to
        project a For example, let's imagine we want to project a social
        network of 'Person' nodes through their association with nodes of
        type 'Institution'. First we match the subgraph, and then make
        the projection:

        '''
        MATCH (p1:Person)-(i:Institution)-(p2:Person)
        PROJECT (p1)-(i)-(p2)
        '''

        In the above example it is important to note the mandatory use of
        the alias style node syntax. To go a step further, let's transfer the
        edges and atributes contained in nodes of type 'City' to neighboring
        nodes of type 'Person', and then project the same social network of
        'Person' nodes through their association with nodes of type
        'Institution'. The query is as follows:

        '''
        MATCH (c:City)-(p1:Person)-(i:Institution)-(p2:Person)
        TRANSFER (c)-(p1)
        PROJECT (p1)-(i)-(p2)
        '''
        ^ In an undirected graph "TRANSFER (c)-(p1)" finds p2 as well.

        And we can keep making up examples:

        '''
        MATCH (p1:Person)-(c:City)-(i:Institution)-(p2:Person)
        MERGE (c)-(i)
        PROJECT (p1)-(i)-(p2)
        '''
        ...

        CURRENTLY ProjX only allows ***1 match per query***! Also, in
        the future it will probably allow soft pattern matching to match
        partial patterns as well.

        Full graph queries:
        -------------------
        To perform an projection over the whole graph and return a modified
        copy, simply tell ProjX what you want to do by combining a verb and
        a simple pattern. Node Type aliases are not necessary for one-line
        queries UNLESS you are using the same node type multiple times in the
        pattern. For example, to transfer all the attributes of nodes of
        type 'Foo' to their neighboring nodes of type 'Bar' and delete the
        'Foo' nodes we can say:

        "TRANSFER (Foo)-(Bar)".

        We can also use a wildcard node type when we want a more flexible
        traversal. Let's project an association graph of people connected
        to other people through a node of any other type:

        "PROJECT (p1:Person)-()-(p2:Person)".

        Here it is important to remember this projection will delete the
        wildcard nodes, but any other nodes that are not matched by this
        pattern remain in the graph. If you only want the returned graph
        to contain the 'Person' nodes, you will need to use a matched
        subgraph queries as defined in the preceding section.

        We can still write multi-line queries that act over the whole graph
        too.

        '''
        TRANSFER (i:Institution)-(p:Person)-(c:City)
        MERGE (i)-(p)
        '''

        And we can continue:

        '''
        MERGE (Foo)-(Bar)
        ...
        '''


        Predicates:
        -----------
        ProjX doesn't currently support predicates such as "AS", "WHERE",
        but it will soon.

        :param query: String. A ProjX query.
        :returns: networkx.Graph. The graph or subgraph with the required
                  schema modfications.
        """  
        clauses = self.parser.parseString(query)
        verb = clauses[0]['verb']
        obj = clauses[0].get('object', '')
        pattern = clauses[0]['pattern']
        mp = _MatchPattern(pattern)  # Fix pattern processor
        paths = self._match(mp)
        if verb == 'match':
            graph = self.match(paths)
        elif verb in ['transfer', 'merge', 'project']:
            graph = self.graph.copy()
            action = self._actions[verb]
            graph, paths = action(graph, paths, mp, pattern, obj)
        else:
            raise SyntaxError('Expected statement to begin with '
                              '"MATCH" "TRANSFER" or "PROJECT".')
        for clause in clauses[1:]:
            verb = clause['verb']
            obj = clause.get('object', '')
            pattern = clause['pattern']
            action = self.actions.get(verb, '')
            if action:
                graph, paths = action(graph, paths, mp, pattern, obj)
            else:
                raise SyntaxError('Expected statement to begin with '
                                  '"TRANSFER" or "PROJECT".')
        graph.remove_nodes_from(self._removals)
        self._removals = set()
        return graph

    def match(self, paths):
        """
        Converts _match output to a networkx.Graph.

        :param path: List of lists. The paths matched
                     by the _match method.
        :returns: networkx.Graph. A matched subgraph.
        """
        # Return a graph comprised by the matched nodes.
        return self.build_subgraph(paths)

    def _match(self, pattern):
        """
        Executes traversals to perform initial match on pattern.

        :param pattern: String. A valid pattern string.
        :returns: List of lists. The matched paths.
        """
        node_type_seq = pattern.node_type_seq
        edge_type_seq = pattern.edge_type_seq
        # Get type sequence and start node type.
        start_type = node_type_seq[0]
        # Store the results of the upcoming traversals.
        path_list = []
        for node, attrs in self.graph.nodes(data=True):
            if attrs[self.node_type] == start_type or not start_type:
                # Traverse the graph using the type sequence
                # as a criteria for a valid path.
                paths = self.traverse(node, node_type_seq[1:], edge_type_seq)
                path_list.append(paths)
        paths = list(chain.from_iterable(path_list))
        if not paths:
            raise Exception('There are no nodes matching '
                            'the given type sequence. Check for '
                            'spelling errors and syntax')
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
        paths = self._match(mp)
        graph, paths = self._project(self.graph.copy(), paths, mp)
        return graph

    def _project(self, graph, paths, mp, pattern=None, obj=None):
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
        source, target = _get_source_target(paths, mp, pattern)
        for path in paths:
            remove = path[source + 1:target]
            self._removals.update(remove)
            graph.add_edge(path[source], path[target])
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
        paths = self._match(mp)
        graph, paths = self._transfer(self.graph.copy(), paths, mp)
        return graph

    def _transfer(self, graph, paths, mp, pattern=None, obj=None):
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
        source, target = _get_source_target(paths, mp, pattern)
        for path in paths:
            # Node type to be transfered.
            transfer_source = path[source]
            # Node type to recieve transfered
            # node attributes.
            transfer_target = path[target]
            # The difference between MERGE and TRANSFER.
            if obj == 'edges' or obj != 'attrs':
                edges = graph[transfer_source]
                new_edges = zip([transfer_target] * len(edges), edges)
                graph.add_edges_from(new_edges)
            if obj == 'attrs' or obj != 'edges':
                attrs = graph.node[transfer_source]
                tp = attrs[self.node_type]
                # Allow for attributes "slugs" to
                # be created during transfer for nodes that
                # take on attributes from multiple transfered nodes.
                attr_counter = 1
                # Transfer the attributes to target nodes.
                for k, v in attrs.items():
                    if k not in [self.node_type, 'visited_from']:
                        attname = '{0}_{1}'.format(tp.lower(), k)
                        if (attname in graph.node[transfer_target] and
                                graph.node[transfer_target].get(attname, '') != v):
                            attname = '{0}{1}'.format(attname, attr_counter)
                            attr_counter += 1
                        graph.node[transfer_target][attname] = v
            self._removals.update([transfer_source])
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
                        self.edge_type, ''
                    )
                    attrs = self.graph.node[nbr]
                    # Here check candidate node validity.
                    # Make sure this path hasn't been checked already.
                    # Make sure it matches the type sequence.
                    # Make sure it's not backtracking on same path.
                    # Kind of a nasty if, but I don't want to 
                    # make a method call.
                    if (current not in attrs['visited_from'] and
                            nbr not in stack and
                            (edge_type_attr == edge_type_seq[depth] or
                             edge_type_seq[depth] == '') and 
                            (attrs[self.node_type] == node_type_seq[depth] or
                             node_type_seq[depth] == '')):
                        self.graph.node[nbr]['visited_from'].append(current)
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
            g.add_edges_from(combined_paths)
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


def _get_source_target(paths, mp, pattern):
    """
    Uses _MatchPattern's alias system to perform a pattern match.
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
        self.nodes = pattern['nodes']
        self.edges = pattern['edges']
        self.node_alias = {}
        self.edge_alias = {}
        self.node_type_seq = []
        self.edge_type_seq = []
        for i, node in enumerate(self.nodes):
            node = node[0]
            node_alias = node['alias']
            self.node_alias[node_alias] = i
            tp = node.get('type', '')
            if tp:
                tp = tp[0]
            self.node_type_seq.append(tp)
        for j, edge in enumerate(self.edges):
            if edge:
                edge = edge[0]
                edge_alias = edge['alias']
                self.edge_alias[edge_alias] = i
                tp = edge.get('type', '')
                if tp:
                    tp = tp[0]
            else:
                tp = ''
                self.edge_alias[''] = i
            self.edge_type_seq.append(tp)


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
    return g


def draw_simple(graph):
    """
    Utility function to draw a labeled, colored graph with Matplotlib.

    :param graph: networkx.Graph
    """
    lbls = labels(graph)
    clrs = colors(graph)
    pos = nx.spring_layout(graph)
    return (nx.draw_networkx(graph, pos=pos, node_color=clrs),
            nx.draw_networkx_labels(graph, pos=pos, labels=lbls))


def labels(graph):
    """
    Utility function that aggreates node attributes as
    labels for drawing graph in Ipython Notebook.

    :param graph: networkx.Graph
    :returns: Dict. Nodes as keys, labels as values.
    """
    labels_dict = {}
    for node, attrs in graph.nodes(data=True):
        label = ''
        for k, v in attrs.items():
            if k != 'visited_from':
                label += '{0}: {1}\n'.format(k, v)
        labels_dict[node] = label
    return labels_dict


def edge_labels(graph, edge_type_attr='type'):
    """
    Utility function that aggreates node attributes as
    labels for drawing graph in Ipython Notebook.

    :param graph: networkx.Graph
    :returns: Dict. Nodes as keys, labels as values.
    """
    labels_dict = {}
    for i, j, attrs in graph.edges(data=True):
        label = attrs[edge_type_attr]
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
