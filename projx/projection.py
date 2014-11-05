# -*- coding: utf-8 -*-
import re
from itertools import chain
import networkx as nx


class Projection(object):
    def __init__(self, graph, type_attr='type'):
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
        target There are two types of queries. The first type are multi-line
        queries that act upon a pattern matched subgraph
        of the wrapped graph. The second type consists of one line statements
        that apply schema modifications across the entire wrapped graph.
        Query language specifications are included in the documentation for
        the execute method

        :param graph: An multi-partite (multi-type) instance of
                      networkx.Graph().

        :param type_attr: A string node attribute name that distinguishes
                          between types (modes). Default is 'type'.

        """
        for node in graph.nodes():
            graph.node[node]['visited_from'] = []
        self.graph = graph
        self.type = type_attr
        self._removals = set()
        self._grammar = {}
        # Updates the grammar with the defined rules.
        self._grammar_rules()

    def grammar_wrapper(self, verb):
        """
        Wraps the grammar rules and adds them to the grammar dictionary for
        easyretrivial during query parse.

        :param verb: String. The ProjX verb assiociated with the wrapped
            function.
        """
        def wrapper(fn):
            self._grammar[verb] = fn
        return wrapper

    def _get_grammar(self):
        """
        Return grammar for grammar property.

        :returns: Dict. A dict containing a mapping of grammar verbs to
            methods.
        """
        return self._grammar
    grammar = property(fget=_get_grammar)

    def _grammar_rules(self):
        """
        A series of functions representing the grammar. These are wrapped
        by the grammar wrapper and added to the grammar. Later during
        the parsing and execution phase these are called as pointers
        to the various graph transformation methods (transfer and project).
        """
        @self.grammar_wrapper('project')
        def execute_project(graph, paths, match_pattern, pattern):
            return self._project(graph, paths, match_pattern, pattern)

        @self.grammar_wrapper('transfer')
        def execute_transfer(graph, paths, match_pattern, pattern):
            return self._transfer(graph, paths, match_pattern, pattern)

        @self.grammar_wrapper('merge')
        def execute_merge(graph, paths, match_pattern, pattern):
            return self._merge(graph, paths, match_pattern, pattern)

    def _clear(self, nbunch):
        """
        Used to clear the visited attribute on a bunch of nodes.

        :param nbunch: Iterable. A bunch of nodes.
        """
        for node in nbunch:
            self.graph.node[node]['visited_from'] = []

    def execute(self, query):
        """
        This takes a ProjX query and executes it.
        The ProjX syntax is as follows:

        Verbs:
        ------
            - "MATCH" Matches a pattern of nodes based on type.
            - "MATCH PARTIAL" Like "MATCH", but also matchs a partial pattern.
                              Coming soon...
            - "MERGE" Merges the edges and attributes of nodes of one
                         type across a specified sequence of neighboring nodes
                         to nodes of another type.
            - "TRANSFER" Like "MERGE", but only transfers attributes.
            - "PROJECT" Projects a relationship between nodes of one
                        type across a specified sequence of neighboring nodes.
            - "RETURN" Specify table/graph and nodes to return. Coming soon...

        Patterns:
        ---------

        Nodes:
        ++++++
        Nodes are represented using (). For the minimal syntax, the
        () contains at least a node type specification, this specification
        corresponds to the attribute set at init type_attr: (Type1).
        For longer queries over subgraphs, it is recommended to
        include an alias with the (): (t1:Type1). This allows for
        cleaner code and prevents errors when using complex pattern
        that repeat types.
            - (f:Foo)
            - (b:Bar)

        Edges:
        ++++++
        Currently, only simple undirected edges are permited. They are
        denoted with the hyphen -. Support for edge types and attrs are
        coming soon.

        Patterns:
        +++++++++
        A pattern is a combination of nodes and edges. It creates a
        "type sequence", or a set of criteria that determain a legal
        path during graph traversal based on node's type_attr. For
        example, if we want to locate all nodes with type_attr == 'Type1'
        that are connected to nodes with type_attr == 'Type2', the pattern
        would be specified as "(t1:Type1)-(t2:Type2)". A pattern can be as
        long as necessary, and can repeat elements. Note that the traversal
        does not permit cycles.
            - "(f1:Foo)-(b:Bar)-(f2:Foo)"
            - "(d:Dog)-(p1:Person)-(p2:Person)-(c:Cat)"

        Queries:
        --------
        ProjX queries combine a verb with a pattern to perform some kind
        of search or schema modification over the graph. First there are
        the multi-line queries. They operate over a matched subgraph and
        can perform a sequence of projections. The simplist statments are
        the one line queries. They operate over the entire wrapped graph,
        making one projection at a time.

        Multi-line queries:
        +++++++++++++++++++
        Multi-line queries must begin with a "MATCH" statement. This
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

        One-line queries:
        +++++++++++++++++
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
        to contain the 'Person' nodes, you will need to use a Multi-line
        query as defined in the following section. For a final example, let's
        "TRANSFER" 'Institution' attributes to nodes of type 'City', but only
        when they are connected through a node of type 'Person':

        "TRANSFER (Institution)-(Person)-(City)"

        And we can continue:

        "MERGE (Foo)-(Bar)"
        ...

        Note that single line queries use the first and last specified
        nodes as the source and target of the operation.

        Predicates:
        -----------
        ProjX doesn't currently support predicates such as "AS", "WHERE",
        but it will soon.

        :param query: String. A ProjX query.
        :returns: networkx.Graph. The graph or subgraph with the required
                  schema modfications.
        """
        # Parse the query.
        p = re.compile(r'''
              MATCH\s+[\(\w:\w\)-]+     # Verb + pattern
            | TRANSFER\s+[\(\w:\w\)-]+  # Verb + pattern
            | MERGE\s+[\(\w:\w\)-]+  # Verb + pattern
            | PROJECT\s+[\(\w:\w\)-]+   # Verb + pattern
            ''', re.VERBOSE | re.IGNORECASE)
        statements = p.findall(query)
        # Find the starting point.
        if not statements:
            raise SyntaxError('Invalid query string.')
        verb, pattern = statements[0].split()
        verb = verb.lower()
        mp = _MatchPattern(pattern)
        if len(statements) == 1:
            # One-liners - operate on whole graph.
            paths = self._match(mp)
            if verb == 'match':
                graph = self.match(paths)
            else:
                graph = self.graph.copy()
                # Use the grammar to perform the required projection.
                grammar_fn = self.grammar.get(verb, '')
                if grammar_fn:
                    graph, paths = grammar_fn(graph, paths, mp, pattern)
                else:
                    raise SyntaxError('Expected statement to begin '
                                      'with "MATCH" "TRANSFER", '
                                      'or "PROJECT".')
        elif len(statements) > 1:
            # Multi-line queries - store projected changes on subgraph.
            if verb != 'match':
                raise SyntaxError('Composed queries must begin '
                                  'with valid MATCH statment.')
            paths = self._match(mp)
            graph = self.match(paths)
            for statement in statements[1:]:
                verb, pattern = statement.split()
                grammar_fn = self.grammar.get(verb.lower(), '')
                if grammar_fn:
                    graph, paths = grammar_fn(graph, paths, mp, pattern)
                else:
                    raise SyntaxError('Expected statement to begin with '
                                      '"TRANSFER" or "PROJECT".')
        else:
            raise SyntaxError('Invalid query string.')
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
        type_seq = pattern.type_seq

        # Get type sequence and start node type.
        if len(type_seq) < 2:
            raise SyntaxError('Patterns must be formated using '
                              'Cypher style notation e.g. '
                              'MATCH (Person)-(City) with at '
                              'least two types in the sequence.')
        start_type = type_seq[0]
        if not start_type:
            raise SyntaxError('Patterns must begin with '
                              'a specified type e.g., '
                              'MATCH (Person)-() is valid, '
                              'MATCH ()-(Person) is not.')
        # Store the results of the upcoming traversals.
        path_list = []
        for node, attrs in self.graph.nodes(data=True):
            if attrs[self.type] == start_type:
                # Traverse the graph using the type sequence
                # as a criteria for a valid path.
                paths = self.traverse(node, type_seq[1:])
                path_list.append(paths)
        paths = list(chain.from_iterable(path_list))
        if not paths:
            raise Exception('There are no nodes matching '
                            'the given type sequence. Check for '
                            'spelling errors and syntax')
        return paths

    def project(self, match_pattern):
        """
        Performs match, executes _project, and returns graph. This can be
        part of programmatic API.

        :param match_pattern: _MatchPattern. The initital pattern specified
                              in "MATCH" statement or in one-line query.
        :returns: networkx.Graph. A projected copy of the wrapped graph
                  or its subgraph.

        """
        paths = self._match(match_pattern)
        graph, paths = self._project(self.graph.copy(), paths, match_pattern)
        return graph

    def _project(self, graph, paths, match_pattern, pattern=None):
        """
        Executes graph "PROJECT" projection.

        :param graph: networkx.Graph. A copy of the wrapped grap or its
                      subgraph.
        :param path: List of lists. The paths matched
                     by the _match method based.
        :param match_pattern: _MatchPattern. The initital pattern specified
                              in "MATCH" statement or in one-line query.
        :param pattern: Optional. String. A valid pattern string. Needed for
                        multi-line query.
        :returns: networkx.Graph. A projected copy of the wrapped graph
                  or its subgraph.
        """
        source, target = _get_source_target(paths, match_pattern, pattern)
        for path in paths:
            remove = path[source + 1:target]
            self._removals.update(remove)
            graph.add_edge(path[source], path[target])
        return graph, paths

    def transfer(self, match_pattern):
        """
        Performs match, executes _transfer, and returns graph. This can be
        part of programmatic API.

        :param match_pattern: _MatchPattern. The initital pattern specified
                              in "MATCH" statement or in one-line query.
        :returns: networkx.Graph. A projected copy of the wrapped graph
                  or its subgraph.
        """
        paths = self._match(match_pattern)
        graph, paths = self._transfer(self.graph.copy(), paths, match_pattern)
        return graph

    def merge(self, match_pattern):
        """
        Performs match, executes _merge, and returns graph. This can be
        part of programmatic API.

        :param match_pattern: _MatchPattern. The initital pattern specified
                              in "MATCH" statement or in one-line query.
        :returns: networkx.Graph. A projected copy of the wrapped graph
                  or its subgraph.

        """
        paths = self._match(match_pattern)
        graph, paths = self._project(self.graph.copy(), paths, match_pattern)
        return graph

    def _merge(self, graph, paths, match_pattern, pattern=None):
        """
        Execute a graph "MERGE" projection.

        :param graph: networkx.Graph. A copy of the wrapped grap or its
                      subgraph.
        :param path: List of lists. The paths matched
                     by the _match method based.
        :param match_pattern: _MatchPattern. The initital pattern specified
                              in "MATCH" statement or in one-line query.
        :param pattern: Optional. String. A valid pattern string. Needed for
                        multi-line query.
        :returns: networkx.Graph. A projected copy of the wrapped graph
                  or its subgraph.
        """
        return self._transfer(graph, paths, match_pattern, pattern, True)

    def _transfer(self, graph, paths, match_pattern,
                  pattern=None, edges=False):
        """
        Execute a graph "TRANSFER" projection.

        :param graph: networkx.Graph. A copy of the wrapped grap or its
                      subgraph.
        :param path: List of lists. The paths matched
                     by the _match method based.
        :param match_pattern: _MatchPattern. The initital pattern specified
                              in "MATCH" statement or in one-line query.
        :param pattern: Optional. String. A valid pattern string. Needed for
                        multi-line query.
        :param edges: Bool. Default False. Settings this to true executes a
                      merge.
        :returns: networkx.Graph. A projected copy of the wrapped graph
                  or its subgraph.
        """
        source, target = _get_source_target(paths, match_pattern, pattern)
        for path in paths:
            # Node type to be transfered.
            transfer_source = path[source]
            # Node type to recieve transfered
            # node attributes.
            transfer_target = path[target]
            # The difference between MERGE and TRANSFER.
            if edges:
                edges = graph[transfer_source]
                new_edges = zip([transfer_target] * len(edges), edges)
                graph.add_edges_from(new_edges)
            attrs = graph.node[transfer_source]
            tp = attrs[self.type]
            # Allow for attributes "slugs" to
            # be created during transfer for nodes that
            # take on attributes from multiple transfered nodes.
            attr_counter = 1
            # Transfer the attributes to target nodes.
            for k, v in attrs.items():
                if k not in [self.type, 'visited_from']:
                    attname = '{0}_{1}'.format(tp.lower(), k)
                    if (attname in graph.node[transfer_target] and
                            graph.node[transfer_target].get(attname, '') != v):
                        attname = '{0}{1}'.format(attname, attr_counter)
                        attr_counter += 1
                    graph.node[transfer_target][attname] = v
            self._removals.update([transfer_source])
        return graph, paths

    def traverse(self, start, type_seq):
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
        max_depth = len(type_seq)
        # When the stack runs out, all candidate
        # nodes have been visited.
        while len(stack) > 0:
            # Traverse!
            if depth < max_depth:
                nbrs = set(self.graph[current])
                for nbr in nbrs:
                    attrs = self.graph.node[nbr]
                    # Here check candidate node validity.
                    # Make sure this path hasn't been checked already.
                    # Make sure it matches the type sequence.
                    # Make sure it's not backtracking on same path.
                    if (current not in attrs['visited_from'] and
                            nbr not in stack and
                            (attrs[self.type] == type_seq[depth] or
                             type_seq[depth] == '')):
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


def _get_source_target(paths, match_pattern, pattern):
    """
    Uses _MatchPattern's alias system to perform a pattern match.
    :param match_pattern: _MatchPattern. The initital pattern specified
                          in "MATCH" statement or in one-line query.
    :param pattern: String. A valid pattern string of aliases.
    """
    if pattern is None:
        source = 0
        target = len(paths[0]) - 1
    else:
        alias_seq = re.findall('\((.*?)\)', pattern)
        try:
            source = match_pattern.alias[alias_seq[0]]
            target = match_pattern.alias[alias_seq[-1]]
        # This is a hack to deal with syntax evolution
        # alias or not in one-liners.
        except KeyError:
            try:
                alias_seq = [a.split(':')[0] for a in alias_seq]
                source = match_pattern.alias[alias_seq[0]]
                target = match_pattern.alias[alias_seq[-1]]
            except:
                raise SyntaxError('Patterns should be formatted either '
                                  '(f:foo)-(b:bar) or (foo)-(bar) '
                                  'for simple queries with more than 1 '
                                  'wildcard node use aliases: '
                                  '"TRANSFER (f:foo)-(wild1:)-(wild2)"')
    return source, target


class _MatchPattern(object):

    def __init__(self, pattern):
        """
        This is a helper class that takes a match pattern and
        maintains an alias dictionary. This allows for multi-line
        queries to utilize alaiases.

        :param :
        """
        self.pattern = pattern  # may not be necessary
        self.alias = {}
        self.type_seq = []
        pattern = re.findall('\((.*?)\)', pattern)
        for i, group in enumerate(pattern):
            if not group:
                tp = ''
                alias = ''
            else:
                group = group.split(':')
                if len(group) == 2:
                    alias, tp = group
                elif len(group) == 1:
                    tp = group[0]
                    alias = tp
                else:
                    raise SyntaxError('Patterns must be formated using '
                                      'Cypher style notation e.g. '
                                      'MATCH (Person)-(City) with at '
                                      'least two types in the sequence.')
            self.type_seq.append(tp)
            self.alias[alias] = i


def test_graph():
    """
    The first tests will use this function I assume.
    :returns: networkx.Graph
    """
    g = nx.Graph([(1, 2), (1, 3), (2, 3), (3, 4), (4, 5), (5, 6),
                  (7, 3), (8, 5), (7, 2), (8, 4), (7, 4), (9, 4),
                  (9, 10), (11, 3)])
    g.node[1] = {'type': 'Person', 'name': 'davebshow'}
    g.node[2] = {'type': 'Institution', 'name': 'western'}
    g.node[3] = {'type': 'City', 'name': 'london'}
    g.node[4] = {'type': 'Institution', 'name': 'the matrix'}
    g.node[5] = {'type': 'City', 'name': 'toronto'}
    g.node[6] = {'type': 'Person', 'name': 'gandalf'}
    g.node[7] = {'type': 'Person', 'name': 'versae'}
    g.node[8] = {'type': 'Person', 'name': 'kreeves'}
    g.node[9] = {'type': 'Person', 'name': 'r2d2'}
    g.node[10] = {'type': 'City', 'name': 'alderon'}
    g.node[11] = {'type': 'Person', 'name': 'curly'}
    return g


def draw_simple(graph):
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


def colors(graph, type_attr='type'):
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
        if attrs[type_attr] not in colors_dict:
            colors_dict[attrs[type_attr]] = float(counter)
            colors.append(float(counter))
            counter += 1
        else:
            colors.append(colors_dict[attrs[type_attr]])
    return colors
