# -*- coding: utf-8 -*-


def neo4j_extractor(extractor_json, graph):
    return {"graph": graph, "query": extractor_json.get("query", "")}


def neo4j_stream(transformers, extractor_context):
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
