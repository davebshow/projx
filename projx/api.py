# -*- coding: utf-8 -*-
from etl import execute_etl
from grammar import parse_query


# DB Style API for the projx/networkx DSL.
def connect(g):
    return Connection(g)


class Connection(object):

    def __init__(self, graph):
        self.graph = graph
        self._cursor = None

    def cursor(self):
        if not self.graph:
            raise Exception("Connection is closed.")
        cursor = Cursor(self)
        self._cursor = cursor
        return cursor

    def commit(self):
        if not getattr(self._cursor, "pending", ""):
            raise Exception("Nothing to commit")
        self.graph = self._cursor.pending
        self._cursor._pending = None
        
    def rollback(self):
        try:
            self._cursor._pending = None
        except AttributeError:
            raise Exception("Nothing to rollback")

    def _reset_cursor(self):
        self._cursor = None

    def close(self):
        self.graph = None
        self._cursor = None

    def __del__(self):
        self.close()


class Cursor(object):

    def __init__(self, connection):
        self.connection = connection
        self.graph = connection.graph
        self._pending = None

    def _get_pending(self):
        return self._pending
    pending = property(fget=_get_pending)

    def execute(self, query):
        if not self.graph:
            raise Exception("Cursor has been closed.")
        etl = parse_query(query)    
        self._pending = execute_etl(etl, self.graph)
        return self._pending

    def close(self):
        self._pending = None
        self.graph = None
        self.connection._reset_cursor()

    def __del__(self):
        self.close()
