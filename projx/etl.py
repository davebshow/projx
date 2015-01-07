# -*- coding: utf-8 -*-
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
        self.extractor = self.extractors[self.extractor_name]

        # Get the loader function.
        self._loaders = {}
        self._init_loaders()
        self.loader = self.loaders[self.loader_name]

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
        def get_nx_loader(transformers, extractor, graph):
            """
            :param tranformers: List of dicts.
            :extractor: function.
            :param graph: networkx.Graph
            :returns: projx.nx_loader
            """
            return nx2nx_loader(transformers, extractor, graph)


# NetworkX Module.
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


def nx2nx_loader(transformers, extractor, graph):
    """
    Loader for NetworkX graph.

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

    :param graph: networkx.Graph
    :param paths: List of lists.
    :returns: networkx.Graph
    """
    removals = set()
    projector = nxprojx.NXProjector(max(graph.nodes()))
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
    projector = nxprojx.NXProjector(max(graph.nodes()))
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