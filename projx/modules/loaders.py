# -*- coding: utf-8 -*-
"""
Loader functions take 5 params: extractor function, stream function,
list of transformers (JSON), the loader JSON, and graph object. They handle
the particulars of extracting the necessary data from the stream, taking into
account the particularities of the source and loading/writing to the
destination.
"""
import networkx as nx
from projx import nxprojx


def nx2nx_loader(extractor, stream, transformers, loader_json, graph):
    """
    Loader for NetworkX graph.

    :returns: networkx.Graph
    """
    extractor_json = extractor(graph)
    graph = extractor_json["graph"]
    node_type_attr = extractor_json["node_type_attr"]
    edge_type_attr = extractor_json["edge_type_attr"]
    if len(transformers) > 1:
        removals = set()
        projector = nxprojx.NXProjector(max(graph.nodes()))
        for trans in stream(transformers, extractor_json):
            record, trans_kwrd, trans = trans
            to_set = trans.get("set", [])
            method = trans.get("method", {"none": []})
            method_kwrd = method.keys()[0]
            params = method.get(method_kwrd, {"args": []})["args"]
            attrs = _nx_lookup_attrs(to_set, record, graph)
            src, target, to_del = _apply_nx2nx_transformer(trans, record)
            fn = projector.transformations[trans_kwrd]
            graph = fn(src, target, graph, attrs, node_type_attr,edge_type_attr,
                       method=method, params=params)
            for i in to_del:
                removals.update([i])
        graph.remove_nodes_from(removals)
    elif len(transformers) == 1:
        graph = nx2nx_single_transform_loader(
            transformers[0],
            extractor_json["paths"],
            graph,
            node_type_attr,
            edge_type_attr
        )
    return graph


def nx2nx_single_transform_loader(transformer, paths, graph, node_type_attr,
                                  edge_type_attr):
    """
    Special transformer/loader for single transformations across nx graphs.

    :param graph: networkx.Graph
    :param paths: List of lists.
    :returns: networkx.Graph
    """
    removals = set()
    projector = nxprojx.NXProjector(max(graph.nodes()))
    trans_kwrd = transformer.keys()[0]
    trans = transformer[trans_kwrd]
    to_set = trans.get("set", [])
    method = trans.get("method", {"none": []})
    method_kwrd = method.keys()[0]
    params = method.get(method_kwrd, {"args": []})["args"]
    fn = projector.transformations[trans_kwrd]
    for record in paths:
        attrs = _nx_lookup_attrs(to_set, record, graph)
        src, target, to_del = _apply_nx2nx_transformer(trans, record)
        graph = fn(src, target, graph, attrs, node_type_attr, edge_type_attr,
                   method=method, params=params)
        for i in to_del:
            removals.update([i])
    graph.remove_nodes_from(removals)
    return graph


def neo4j2nx_loader(extractor, stream, transformers, loader_json, graph):
    output_graph = nx.Graph()
    extractor_json = extractor(graph)
    if len(transformers) > 0:
        for trans in stream(transformers, extractor_json):
            record, trans_kwrd, trans = trans
            to_set = trans.get("set", [])
            attrs = _neo4j_lookup_attrs(to_set, record)
            pattern = trans.get("pattern", [])
            if trans_kwrd == "node":
                try:
                    node = pattern[0].get("node", {})
                    unique = node.get("unique", "")
                    alias = node.get("alias", "")
                    unique_id = record[alias][unique]
                except (IndexError, KeyError):
                    raise Exception("Invalid transformation pattern.")
                if unique_id not in output_graph:
                    output_graph.add_node(unique_id, attrs)
                else:
                    output_graph.node[unique_id].update(attrs)
            elif trans_kwrd == "edge":
                source, target = _neo4j_get_source_target(record, pattern)
                output_graph = nxprojx.project(source, target, output_graph,
                                               method="", attrs=attrs)
    else:
        raise Exception("Please define transformation(s).")
    return output_graph


def neo4j2edgelist_loader(extractor, stream, transformers, loader_json, graph):
    extractor_json = extractor(graph)
    try:
        filename = loader_json["filename"]
    except KeyError:
        raise Exception("Enter valid filename.")
    delim = loader_json.get("delim", ",")
    newline = loader_json.get("newline", "\n")
    if len(transformers) > 0:
        with open(filename, "w") as f:
            for trans in stream(transformers, extractor_json):
                record, trans_kwrd, trans = trans
                pattern = trans.get("pattern", [])
                if trans_kwrd == "edge":
                    source, target = _neo4j_get_source_target(record, pattern)
                    line = "{0}{1}{2}{3}".format(source, delim, target, newline)
                    f.write(line)
    else:
        raise Exception("Please define transformation(s).")


def _apply_nx2nx_transformer(trans, record):
    pattern = trans["pattern"]
    source, target = _nx_get_source_target(pattern, record)
    delete_alias = trans.get("delete", {}).get("alias", [])
    to_delete = [record[alias] for alias in delete_alias]
    return source, target, to_delete


def _nx_get_source_target(pattern, record):
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
    source = record[alias_seq[0]]
    target = record[alias_seq[-1]]
    return source, target


def _nx_lookup_attrs(to_set, record, graph):
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
                node = record[alias]
                value = graph.node[node].get(lookup_key, "")
        attrs[key] = value
    return attrs


def _neo4j_get_source_target(record, pattern):
    try:
        alias_seq = [(p["node"]["alias"], p["node"]["unique"])
                     for p in pattern[0::2]]
        source = record[alias_seq[0][0]][alias_seq[0][1]]
        target = record[alias_seq[-1][0]][alias_seq[-1][1]]
    except KeyError:
        raise Exception("Invalid transformation pattern.")
    return source, target


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
