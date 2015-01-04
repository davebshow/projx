from itertools import cycle, islice
from pyparsing import (Word, alphanums, OneOrMore, ZeroOrMore, Group, 
                       stringEnd, Suppress, Literal, CaselessKeyword,
                       Optional, Forward, quotedString, removeQuotes)


# Used throughout as a variable/attr name.
var = Word(alphanums, "_" + alphanums)

############### MATCH STATEMENT ######################

match = CaselessKeyword("MATCH")
graph = CaselessKeyword("GRAPH") | CaselessKeyword("SUBGRAPH")
graph.setParseAction(lambda t: t[0].lower())

################ Transformations #######################

transformation = (
    CaselessKeyword("TRANSFER") |
    CaselessKeyword("PROJECT") |
    CaselessKeyword("COMBINE") 
)
transformation.setParseAction(lambda t: t[0].lower())

################ NODE AND EDGE PATTERNS ###################

# Used for node and edge patterns.
seperator = Suppress(Literal(":"))
tp = seperator + Word(alphanums, "_" + alphanums)

# Node type pattern.
node_open = Suppress(Literal("("))
node_close = Suppress(Literal(")"))
node_content = var("alias") + Optional(tp("type").setParseAction(lambda t: t[0]))

node = node_open + node_content + node_close

# Edge patterns.
edge_marker = Suppress(Literal("-"))
edge_open = Suppress(Literal("["))
edge_close = Suppress(Literal("]"))
edge_content = (
    edge_open + var("alias") + 
    Optional(tp("type").setParseAction(lambda t: t[0])) + edge_close
)

edge = edge_marker + Optional(edge_content + edge_marker)

# Match/Transformation pattern.
pattern = Forward() 
pattern << node.setResultsName("nodes", listAllMatches=True) + ZeroOrMore(
        edge.setResultsName("edges", listAllMatches=True) + pattern
)  

################### PREDICATE CLAUSES #######################

# Used for the creation of new nodes with combine or project.
new = CaselessKeyword("NEW")
new.setParseAction(lambda t: t[0].lower())

# Comma seperated argument pattern
csv_pattern = Forward()
csv_pattern << var.setResultsName("pattern", listAllMatches=True) + ZeroOrMore(
     Suppress(Literal(",")) + csv_pattern
)

# Getter/Setter Pattern.
left = new | var
attr = Word(alphanums, "." + alphanums)
right = attr("value_lookup") | quotedString("value").setParseAction(removeQuotes)

gttr_sttr = (
    left("alias") + Suppress(Literal(".")) + var("key") + 
    Suppress(Literal("=")) + right
)

pred_pattern = Forward()
pred_pattern << gttr_sttr.setResultsName("pattern", listAllMatches=True) + ZeroOrMore(
    Suppress(Literal(",")) + pred_pattern
)

################# DELETE #####################
delete = CaselessKeyword("DELETE")
delete.setParseAction(lambda t: t[0].lower())
delete_clause = delete("predicate") + csv_pattern

################## SET ########################
setter = CaselessKeyword("SET")
setter.setParseAction(lambda t: t[0].lower())
set_clause = setter("predicate") + pred_pattern

################# METHOD ######################
method = CaselessKeyword("METHOD")
obj = (
    CaselessKeyword("ATTRS").setParseAction(lambda t: t[0].lower()) |
    CaselessKeyword("EDGES").setParseAction(lambda t: t[0].lower())
)
algorithm = CaselessKeyword("JACCARD").setParseAction(lambda t: t[0].lower())
over = CaselessKeyword("OVER").setParseAction(lambda t: t[0].lower())
projection_clause = algorithm("algo") + over + csv_pattern("over")  
method_clause = method("predicate") + (obj | projection_clause)("pattern")

# Allows for one use of each predicate verb.
predicate_clause = (
    Optional(delete_clause("delete")) & Optional(set_clause("set")) & 
    Optional(method_clause("method"))
)

###################### QUERY ###########################
match_clause = match("match") + Optional(graph("graph")) + pattern("match_pattern")

transformation_clause = (
    transformation("transformation") + pattern("transform_pattern") + 
    Optional(predicate_clause)
)

transform_pattern = Forward()
transform_pattern << transformation_clause.setResultsName(
    "transformations", listAllMatches=True) + ZeroOrMore(transform_pattern)

parser = match_clause + Optional(transform_pattern) + stringEnd


def parse_query(query):
    """
    Parses grammar output to ETL JSON.

    :param query: Str. DSL query.
    :returns: JSON. ETL JSON.
    """
    prsd = parser.parseString(query)
    match_pattern = prsd["match_pattern"]
    nodes = [{"node": node.asDict()} for node in match_pattern["nodes"]]
    edges = [{"edge": edge.asDict()} for edge in match_pattern["edges"]]
    transformations = prsd.get("transformations", [])
    etl = {
        "extractor": {
            "networkx": {
                "type": prsd.get("graph", "subgraph"),
                "traversal": list(roundrobin(nodes, edges))
            }
        },
        "transformers": map(parse_transformation, transformations),
        "loader": {
            "networkx": {}
        }
    }
    return etl 


def parse_transformation(transformation):
    """
    :param transformations: List. Parser output.
    :returns: List of dicts. Transformers.
    """
    to_set = []
    to_delete = []
    over = {}
    algorithm = "none"
    trans = transformation["transformation"]
    trans_pattern = transformation["transform_pattern"]
    trans_nodes = [{"node": node.asDict()} for node in trans_pattern["nodes"]]
    trans_edges = [{"edge": edge.asDict()}for edge in trans_pattern["edges"]]
    setter = transformation.get("set", "")
    if setter:
        to_set = [s.asDict() for s in setter["pattern"]]
    delete = transformation.get("delete", "")
    if delete:
        to_delete = delete["pattern"].asList()
    method = transformation.get("method", "")
    if method:
        algorithm = method["algo"]
        if algorithm == "jaccard":
            over = method["over"].asList()
    transformer = {
        trans: {
            "method": {algorithm: {"over": over}},
            "pattern": list(roundrobin(trans_nodes, trans_edges)),
            "set": to_set,
            "delete": {"alias": to_delete}
        }
    }
    return transformer


def roundrobin(*iterables):
    "roundrobin('ABC', 'D', 'EF') --> A D E B F C"
    # Recipe credited to George Sakkis
    pending = len(iterables)
    nexts = cycle(iter(it).next for it in iterables)
    while pending:
        try:
            for next in nexts:
                yield next()
        except StopIteration:
            pending -= 1
            nexts = cycle(islice(nexts, pending))
