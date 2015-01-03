# -*- coding: utf-8 -*-
from etl import execute_etl
from grammar import parse_query


# DB Style API for the projx/networkx DSL.
def connect(g):
    return Connection(g)


class Connection(object):

    def __init__(self, graph):
        self.graph = graph
        self._cursor = Cursor(self.graph)

    def cursor(self):
        return self._cursor

    def commit(self):
        self.graph = self._cursor.graph

    def rollback(self):
        self._cursor.graph = self.graph


class Cursor(object):

    def __init__(self, graph):
        self.graph = graph

    def execute(self, query):
        etl = parse_query(query)    
        self.graph = execute_etl(etl, self.graph)
        return self.graph