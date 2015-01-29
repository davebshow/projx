# -*- coding: utf-8 -*-
import networkx as nx
from nxxtrct import _nx_lookup_attrs
from projx import nxprojx


def nx2nx_loader(extractor, stream, transformers, loader_json, graph):
    """
    Loader for NetworkX graph.

    :returns: networkx.Graph
    """
    extractor_context = extractor(graph)
    graph = extractor_context["graph"]
    node_type_attr = extractor_context["node_type_attr"]
    edge_type_attr = extractor_context["edge_type_attr"]
    if len(transformers) > 1:
        removals = set()
        projector = nxprojx.NXProjector(max(graph.nodes()))
        for trans in stream(transformers, extractor_context):
            record, trans_kwrd, trans, attrs = trans
            src, target, to_del, method, params = _parse_nx2nx_transformer(
                trans, record
            )
            fn = projector.transformations[trans_kwrd]
            graph = fn(src, target, graph, attrs, node_type_attr,edge_type_attr,
                       method=method, params=params)
            for i in to_del:
                removals.update([i])
        graph.remove_nodes_from(removals)
    elif len(transformers) == 1:
        graph = nx2nx_transform_and_load(
            transformers[0],
            extractor_context["paths"],
            graph,
            node_type_attr,
            edge_type_attr
        )
    return graph


def nx2nx_transform_and_load(transformer, paths, graph, node_type_attr,
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
    fn = projector.transformations[trans_kwrd]
    for record in paths:
        attrs = _nx_lookup_attrs(to_set, record, graph)
        src, target, to_del, method, params = _parse_nx2nx_transformer(
            trans, record
        )
        graph = fn(src, target, graph, attrs, node_type_attr, edge_type_attr,
                   method=method, params=params)
        for i in to_del:
            removals.update([i])
    graph.remove_nodes_from(removals)
    return graph


def _parse_nx2nx_transformer(trans, record):
    pattern = trans["pattern"]
    source, target = _nx_get_source_target(pattern, record)
    delete_alias = trans.get("delete", {}).get("alias", [])
    to_delete = [record[alias] for alias in delete_alias]
    method = trans.get("method", {"none": []})
    method_kwrd = method.keys()[0]
    params = method.get(method_kwrd, {"args": []})["args"]
    return source, target, to_delete, method_kwrd, params


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


def neo4j2nx_loader(transformers, extractor, loader_json, graph):
    output_graph = nx.Graph()
    query = extractor().get("query", "")
    if len(transformers) > 0 and query:
        for trans in neo4j_transformer(query, transformers, graph):
            record, trans_kwrd, trans, attrs = trans
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
    return output_graph


def neo4j2edgelist_loader(transformers, extractor, loader, graph):
    query = extractor().get("query", "")
    try:
        filename = loader["filename"]
    except KeyError:
        raise Exception("Enter valid filename.")
    delim = loader.get("delim", ",")
    newline = loader.get("newline", "\n")
    if len(transformers) > 0 and query:
        with open(filename, "w") as f:
            for trans in neo4j_transformer(query, transformers, graph):
                record, trans_kwrd, trans, _ = trans
                pattern = trans.get("pattern", [])
                if trans_kwrd == "edge":
                    source, target = _neo4j_get_source_target(record, pattern)
                    line = "{0}{1}{2}{3}".format(source, delim, target, newline)
                    f.write(line)
    else:
        raise Exception("Please define query and transformation(s).")


def _neo4j_get_source_target(record, pattern):
    try:
        alias_seq = [(p["node"]["alias"], p["node"]["unique"])
                     for p in pattern[0::2]]
        source = record[alias_seq[0][0]][alias_seq[0][1]]
        target = record[alias_seq[-1][0]][alias_seq[-1][1]]
    except KeyError:
        raise Exception("Invalid transformation pattern.")
    return source, target
