# -*- coding: utf-8 -*-
"""
Extractor and stream generator for the Neo4j data source.
"""
try:
    from py2neo import Graph
except ImportError:
    print("Install py2neo to use Neo4j module.")

def neo4j_extractor(extractor_json, graph):
    if isinstance(graph, str):
        graph = Graph(graph)
    extractor_json.update({"graph": graph})
    return extractor_json


def neo4j_stream(transformers, extractor_json):
    graph = extractor_json["graph"]
    query = extractor_json["query"]
    for record in graph.cypher.stream(query):
        for transformer in transformers:
            trans_kwrd = transformer.keys()[0]
            trans = transformer[trans_kwrd]
            yield record, trans_kwrd, trans
