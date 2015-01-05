# -*- coding: utf-8 -*-
import projector


# ETL can be extended beyond NetworkX.
def execute_etl(etl, graph):
    """
    Main API function. Executes ETL on graph. 

    :param etl: ETL JSON.
    :param graph: The source graph.
    :return graph: The projected graph.
    """
    etl = _ETL(etl)
    # Extractor is a function that returns a projector class.
    extractor = etl.extractor
    # List of transformers.
    transformers = etl.transformers
    # Loader can be a class or function that contains transformer.
    loader = etl.loader
    # projector is a class that implements the match, _project, _transform,
    # and _combine methods.
    projector = extractor(graph)
    # Return paths or stream to be passed to loader transformer.
    paths = projector.match()
    # Loader should accept a transfomer list, a projector object, and the
    # paths generator.
    graph = loader(transformers, projector, paths)
    return graph


class _ETL(object):

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
        def get_nx_loader(transformers, projector, paths):
            """
            :param tranformers: List of dicts.
            :projector: class with base projx.Baseprojector
            :param paths: List of lists.
            :returns: projx.nx_loader
            """
            return nx_loader(transformers, projector, paths)


# NetworkX Module.
def nx_loader(transformers, projector, paths):
    """
    Loader for NetworkX graph.

    :param projector: projx.NXprojector
    :param paths: List of lists.
    :returns: networkx.Graph
    """
    if projector.proj_type != "graph":
        projector.graph = projector.build_subgraph(paths)
    graph = projector.graph
    if len(transformers) > 1: 
        graph = nx_transformer_pipeline(transformers, projector, graph, paths)
    elif len(transformers) == 1:
        graph = nx_transformer(transformers, projector, graph, paths)
    return graph


def nx_transformer(transformers, projector, graph, paths):
    """
    Static transformer for NetworkX graph. Single transformation.

    :param projector: projx.NXprojector
    :param graph: networkx.Graph
    :param paths: List of lists.
    :returns: networkx.Graph
    """
    removals = set()
    transformer = transformers[0]
    trans_kwrd = transformer.keys()[0]
    trans = transformer[trans_kwrd]
    pattern = trans["pattern"]
    source, target = _get_source_target(projector.node_alias, pattern)
    to_set = trans.get("set", [])
    fn = projector.transformations[trans_kwrd]
    delete_alias = trans.get("delete", {}).get("alias", [])
    to_delete = [projector.node_alias[alias] for alias in delete_alias]
    method = trans.get("method", {})
    for path in paths:
        source_node = path[source]
        target_node = path[target]
        # Extract to function
        attrs = _get_attrs(projector.node_alias, graph, to_set, path)
        graph = fn(source_node, target_node, attrs, graph, method=method)
        for i in to_delete:
            removals.update([path[i]])
    graph.remove_nodes_from(removals)
    return graph


def nx_transformer_pipeline(transformers, projector, graph, paths):
    """
    Pipeline transformer for NetworkX graph. Multiple transformations.

    :param transformers: List.
    :param projector: projx.NXprojector
    :param graph: networkx.Graph
    :param paths: List of lists.
    :returns: networkx.Graph
    """
    removals = set()
    for path in paths:
        for transformer in transformers:
            trans_kwrd = transformer.keys()[0]
            trans = transformer[trans_kwrd]
            pattern = trans["pattern"]
            source, target = _get_source_target(projector.node_alias, pattern)
            source_node = path[source]
            target_node = path[target]
            to_set = trans.get("set", [])
            method = trans.get("method", {})
            attrs = _get_attrs(projector.node_alias, graph, to_set, path)
            fn = projector.transformations[trans_kwrd]
            graph = fn(source_node, target_node, attrs, graph, method=method)
            delete_alias = trans.get("delete", {}).get("alias", [])
            to_delete = [projector.node_alias[alias] for alias in delete_alias]
            for i in to_delete:
            	removals.update([path[i]])
    graph.remove_nodes_from(removals)
    return graph


def nx_extractor(extractor, graph):
    node_type_attr = extractor.get("node_type_attr", "type")
    edge_type_attr = extractor.get("edge_type_attr", "type")
    traversal = extractor.get("traversal", [])
    nodes = traversal[0::2]
    edges = traversal[1::2]
    proj_type = extractor.get("type", "subgraph")
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
    query = (node_type_seq, edge_type_seq)
    return projector.NXProjector(graph, proj_type, query, node_alias,
                                   edge_alias, node_type_attr, edge_type_attr)


def _get_attrs(node_alias, graph, to_set, path):
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
