# -*- coding: utf-8 -*-
"""
Main API functions.
"""
from etl import ETL
from grammar import parse_query


def execute_etl(etl, graph=None):
    """
    Main API function. Executes ETL on graph.

    :param etl: ETL JSON.
    :param graph: The source graph.
    :return graph: The projected graph.
    """
    etl = ETL(etl)
    if graph is None:
        graph = etl.extractor_json[etl.extractor_name].get("uri", "")
    # Extractor is a function that returns the data source and all necessary
    # info to open up a data stream.
    extractor = etl.extractor
    # Yields four objects that are processed by transformers in load.
    # Each stream is custom for an extractor, but the output is generic
    # for specified loaders
    stream = etl.stream
    # List of transformers.
    transformers = etl.transformers
    # Loader can be a class or function that runs the etl pipeline.
    loader = etl.loader
    # Loaders accept extractor, stream, transformers, graph
    graph = loader(extractor, stream, transformers, graph)
    return graph


class Projection(object):

    def __init__(self, graph, node_type_attr="type", edge_type_attr="type"):
        """
        Main API class for the projx DSL.

        :param graph: networkx.Graph
        """
        self._graph = graph
        self._node_type_attr = node_type_attr
        self._edge_type_attr = edge_type_attr

    def execute(self, query):
        """
        Execute a query written in the projx DSL.

        :param query: Str. projx DSL query.
        :returns: networkx.Graph
        """
        etl = parse_query(query)
        etl["extractor"]["networkx"]["node_type_attr"] = self._node_type_attr
        etl["extractor"]["networkx"]["edge_type_attr"] = self._edge_type_attr
        return execute_etl(etl, self._graph)
