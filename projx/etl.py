# -*- coding: utf-8 -*-
import networkx as nx

import nxprojx


class ETL(object):

    def __init__(self, etl):
        """
        A helper class that parses the ETL JSON, and initializes the
        required extractor, transformers, and loader. Also store key
        variables used in transformation/loading.

        :param etl: ETL JSON.
        """
        # Get the extractor info.
        try:
            self._extractor = etl["extractor"]
            self.extractor_name = self._extractor.keys()[0]
        except (KeyError, IndexError):
            raise Exception("Please define valid extractor")

        # Get the transformers.
        self.transformers = etl.get("transformers", [])

        # Get the loader info.
        try:
            self._loader = etl["loader"]
            self.loader_name = self._loader.keys()[0]
        except (KeyError, IndexError):
            raise Exception("Please define valid loader.")

        # Get the extractor function.
        self._extractors = {}
        self._init_extractors()
        try:
            self.extractor = self.extractors[self.extractor_name]
        except KeyError:
            raise Exception(
                "{0} extractor not implemented".format(self.extractor_name)
            )

        # Get the loader function.
        self._loaders = {}
        self._init_loaders()
        try:
            self.loader = self.loaders[self.loader_name]
        except KeyError:
            raise Exception(
                "{0} loader not implemented".format(self.loader_name)
            )

    def _get_extractors(self):
        return self._extractors
    extractors = property(fget=_get_extractors)

    def extractors_wrapper(self, extractor):
        """
        :param extractor: Str.
        """
        def wrapper(fn):
            self.extractors[extractor] = fn
        return wrapper

    def _init_extractors(self):
        """
        Update extractors dict to allow for extensible extractor
        functionality. Here we set the query attribute that will be passed to
        extractor match method. Also, each extractor function will return
        a projector class object that defines a match method.
        """
        @self.extractors_wrapper("networkx")
        def get_nx_extractor(graph):
            """
            :param graph: networkx.Graph
            :returns: projx.nx_extractor
            """
            return nx_extractor(self._extractor[self.extractor_name], graph)

        @self.extractors_wrapper("neo4j")
        def get_neo4j_extractor():
            """
            :returns: projx.nx_extractor
            """
            return neo4j_extractor(self._extractor[self.extractor_name])

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
        @self.loaders_wrapper("nx2nx")
        def get_nx2nx_loader(transformers, extractor, graph):
            """
            :param tranformers: List of dicts.
            :extractor: function.
            :param graph: networkx.Graph
            :returns: projx.nx_loader
            """
            return nx2nx_loader(transformers, extractor,
                                self._loader[self.loader_name], graph)

        @self.loaders_wrapper("neo4j2nx")
        def get_neo4j2nx_loader(transformers, extractor, graph):
            """
            :param tranformers: List of dicts.
            :extractor: function.
            :param graph: networkx.Graph
            :returns: projx.nx_loader
            """
            return neo4j2nx_loader(transformers, extractor,
                                   self._loader[self.loader_name], graph)


        @self.loaders_wrapper("neo4j2edgelist")
        def get_neo4j2edgelist_loader(transformers, extractor, graph):
            """
            :param tranformers: List of dicts.
            :extractor: function.
            :param graph: networkx.Graph
            :returns: projx.nx_loader
            """
            return neo4j2edgelist_loader(transformers, extractor,
                                   self._loader[self.loader_name], graph)


########## Loaders ###########

def neo4j2nx_loader(transformers, extractor, loader, graph):
    output_graph = nx.Graph()
    query = extractor().get("query", "")
    if len(transformers) > 0 and query:
        for trans in neo4j_transformer(query, transformers, graph):
            record, trans_kwrd, trans, attrs = trans
            pattern = trans.get("pattern", [])
            if trans_kwrd == "node":
                try:
                    node = pattern[0].get("node", {})
                    unique = node.get("unique", "")
                    alias = node.get("alias", "")
                    unique_id = record[alias][unique]
                except (IndexError, KeyError):
                    raise Exception("Invalid transformation pattern.")
                if unique_id not in output_graph:
                    output_graph.add_node(unique_id, attrs)
                else:
                    output_graph.node[unique_id].update(attrs)
            elif trans_kwrd == "edge":
                source, target = _neo4j_get_source_target(record, pattern)
                output_graph = nxprojx.project(source, target, output_graph,
                                               method="", attrs=attrs)
    return output_graph


def neo4j2edgelist_loader(transformers, extractor, loader, graph):
    output_graph = nx.Graph()
    query = extractor().get("query", "")
    try:
        filename = loader["filename"]
    except KeyError:
        raise Exception("Enter valid filename.")
    delim = loader.get("delim", ",")
    newline = loader.get("newline", "\n")
    if len(transformers) > 0 and query:
        with open(filename, "w") as f:
            for trans in neo4j_transformer(query, transformers, graph):
                record, trans_kwrd, trans, attrs = trans
                pattern = trans.get("pattern", [])
                if trans_kwrd == "edge":
                    source, target = _neo4j_get_source_target(record, pattern)
                    line = "{0}{1}{2}{3}".format(source, delim, target, newline)
                    f.write(line)
    else:
        raise Exception("Please define query and transformation(s).")
            

def _neo4j_get_source_target(record, pattern):
    try:
        alias_seq = [(p["node"]["alias"], p["node"]["unique"]) 
                     for p in pattern[0::2]]
        source = record[alias_seq[0][0]][alias_seq[0][1]]
        target = record[alias_seq[-1][0]][alias_seq[-1][1]]
    except KeyError:
        raise Exception("Invalid transformation pattern.")
    return source, target


def nx2nx_loader(transformers, extractor, loader, graph):
    """
    Loader for NetworkX graph.

    :returns: networkx.Graph
    """
    context = extractor(graph)
    graph = context["graph"]
    if len(transformers) > 1:
        removals = set()
        projector = nxprojx.NXProjector(max(graph.nodes()))
        for trans in nx_transformer(transformers, graph, context["paths"], 
                                    context["node_alias"]):
            path, trans_kwrd, trans, node_alias, attrs = trans
            src, target, to_del, method, params = _parse_nx2nx_transformer(
                trans, node_alias
            )
            fn = projector.transformations[trans_kwrd]
            graph = fn(path[src], path[target], graph, attrs, context["node_type_attr"],
                       context["edge_type_attr"], method=method, params=params)
            for i in to_del:
                removals.update([path[i]])
        graph.remove_nodes_from(removals)
    elif len(transformers) == 1:
        # Repeats code, but represents the most common nx use case.
        graph = nx2nx_transform_and_load(
            transformers[0],
            graph,
            context["paths"],
            context["node_alias"],
            context["node_type_attr"],
            context["edge_type_attr"]
        )
    return graph


def nx2nx_transform_and_load(transformer, graph, paths, node_alias,
                             node_type_attr, edge_type_attr):
    """
    Special transformer/loader for single transformations across nx graphs.

    :param graph: networkx.Graph
    :param paths: List of lists.
    :returns: networkx.Graph
    """
    removals = set()
    projector = nxprojx.NXProjector(max(graph.nodes()))
    trans_kwrd = transformer.keys()[0]
    trans = transformer[trans_kwrd]
    to_set = trans.get("set", [])
    src, target, to_del, method, params = _parse_nx2nx_transformer(
        trans, node_alias
    )
    fn = projector.transformations[trans_kwrd]
    for path in paths:
        attrs = _nx_lookup_attrs(node_alias, graph, to_set, path)
        graph = fn(path[src], path[target], graph, attrs, node_type_attr,
                   edge_type_attr, method=method, params=params)
        for i in to_del:
            removals.update([path[i]])
    graph.remove_nodes_from(removals)
    return graph


def _parse_nx2nx_transformer(trans, node_alias):
    pattern = trans["pattern"]
    source, target = _nx_get_source_target(node_alias, pattern)
    delete_alias = trans.get("delete", {}).get("alias", [])
    to_delete = [node_alias[alias] for alias in delete_alias]
    method = trans.get("method", {"none": []})
    method_kwrd = method.keys()[0]
    params = method.get(method_kwrd, {"args": []})["args"]
    return source, target, to_delete, method_kwrd, params


def _nx_get_source_target(node_alias, pattern):
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


########## Neo4j Module ##########

def neo4j_extractor(extractor):
    return {"query": extractor.get("query", "")}


def neo4j_transformer(query, transformers, graph):
    for record in graph.cypher.stream(query):
        for transformer in transformers:
            trans_kwrd = transformer.keys()[0]
            trans = transformer[trans_kwrd]
            to_set = trans.get("set", [])
            attrs = _neo4j_lookup_attrs(to_set, record)
            yield record, trans_kwrd, trans, attrs


def _neo4j_lookup_attrs(to_set, record):
    attrs = {}
    for i, attr in enumerate(to_set):
        key = attr.get("key", i)
        value = attr.get("value", "")
        if not value:
            lookup = attr.get("value_lookup", "")
            if lookup:
                alias, lookup_key = lookup.split(".")
                node = record[alias]
                value = node[lookup_key]
        attrs[key] = value
    return attrs


########## NetworkX module ##########
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
    graph = nxprojx.reset_index(graph)
    paths = nxprojx.match(node_type_seq, edge_type_seq, graph, node_type_attr,
                          edge_type_attr)
    if proj_type != "graph":
        graph = nxprojx.build_subgraph(paths, graph)
    return {
        "graph": graph,
        "paths": paths,
        "node_alias": node_alias,
        "edge_alias": edge_alias,
        "node_type_attr": node_type_attr,
        "edge_type_attr": edge_type_attr
    }


def nx_transformer(transformers, graph, paths, node_alias):
    """
    Pipeline transformer for NetworkX graph. Multiple transformations.

    :param transformers: List.
    :param projector: projx.NXprojector
    :param graph: networkx.Graph
    :param paths: List of lists.
    :returns: networkx.Graph
    """
    for path in paths:
        for transformer in transformers:
            trans_kwrd = transformer.keys()[0]
            trans = transformer[trans_kwrd]
            to_set = trans.get("set", [])
            attrs = _nx_lookup_attrs(node_alias, graph, to_set, path)
            yield path, trans_kwrd, trans, node_alias, attrs


def _nx_lookup_attrs(node_alias, graph, to_set, path):
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
                value = graph.node[node].get(lookup_key, "")
        attrs[key] = value
    return attrs
