# -*- coding: utf-8 -*-
import exceptions
from etl import execute_etl
from grammar import parse_query


# DB Style API for the projx/networkx DSL.
def connect(g):
    return Connection(g)


class Connection(object):

    def __init__(self, graph):
        """
        :param graph: networkx.Graph
        """
        self.graph = graph
        self._cursor = None

    def cursor(self):
        """
        :returns: projx.Cursor
        """
        cursor = Cursor(self)
        self._cursor = cursor
        return cursor

    def commit(self):
        if not getattr(self._cursor, "pending", ""):
            raise Warning("Nothing to commit")
        self.graph = self._cursor.pending
        self._cursor._pending = None
        
    def rollback(self):
        if not getattr(self._cursor, "pending", ""):
            raise Warning("Nothing to rollback")
        self._cursor._pending = None

    def _reset_cursor(self):
        self._cursor = None


class Cursor(object):

    def __init__(self, connection):
        self.connection = connection
        self.graph = connection.graph
        self._pending = None

    def _get_pending(self):
        """
        :returns: networkx.Graph
        """
        return self._pending
    pending = property(fget=_get_pending)

    def execute(self, query):
        """
        :param query: Str. projx DSL query.
        :returns: networkx.Graph
        """
        etl = parse_query(query)    
        self._pending = execute_etl(etl, self.graph)
        return self._pending
