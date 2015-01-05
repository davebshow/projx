# -*- coding: utf-8 -*-
from etl import execute_etl
from grammar import parse_query


def project(g):
    """
    Constructor function for the Projection API.

    :param graph: networkx.Graph
    :returns: projx.Projection
    """
    return Projection(g)


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
