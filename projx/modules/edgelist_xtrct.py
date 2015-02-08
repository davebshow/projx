# -*- coding: utf-8 -*-
"""
Extractor and stream generator for the bare edgelist data source.
"""
from projx import nxprojx


def edgelist_extractor(extractor_json, graph):
    filename = extractor_json.get("filename", "")
    pattern = extractor_json.get("pattern", [])
    try:
        alias = [n["node"]["alias"] for n in pattern[0::2]]
    except KeyError:
        raise Exception("Invalid pattern")
    extractor_json.update({"alias": alias})
    if not filename:
        raise Exception("Unspecified source file")
    return extractor_json


def edgelist_stream(transformers, extractor_json):
    filename = extractor_json["filename"]
    delim = extractor_json.get("delim", ",")
    with open(filename, "rb") as f:
        for line in f:
            line = line.rstrip("\n")
            record = nxprojx.Record(line.split(delim)[:2], extractor_json["alias"])
            for transformer in transformers:
                trans_kwrd = transformer.keys()[0]
                trans = transformer[trans_kwrd]
                to_set = trans.get("set", [])
                attrs = _edgelist_value(to_set)
                yield record, trans_kwrd, trans, attrs


def _edgelist_value(to_set):
    attrs = {}
    for i, attr in enumerate(to_set):
        key = attr.get("key", i)
        value = attr.get("value", "")
        attrs[key] = value
    return attrs
