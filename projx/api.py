# -*- coding: utf-8 -*-
from etl import ETL
from grammar import parse_query


class Projection(object):

    def __init__(self, graph):
        """
        Main API class for the projx DSL.

        :param graph: networkx.Graph
        """
        self._graph = graph

    def execute(self, query):
        """
        Execute a query written in the projx DSL.

        :param query: Str. projx DSL query.
        :returns: networkx.Graph
        """
        return execute_etl(parse_query(query), self._graph)


# ETL can be extended beyond NetworkX.
def execute_etl(etl, graph):
    """
    Main API function. Executes ETL on graph.

    :param etl: ETL JSON.
    :param graph: The source graph.
    :return graph: The projected graph.
    """
    etl = ETL(etl)
    # Extractor is a function that returns a projector class.
    extractor = etl.extractor
    # List of transformers.
    transformers = etl.transformers
    # Loader can be a class or function that contains transformer.
    loader = etl.loader
    # Loader should accept a transfomer list, a projector object, and the
    # graph.
    graph = loader(transformers, extractor, graph)
    return graph
