from itertools import cycle, islice
from pyparsing import (Word, alphanums, OneOrMore, ZeroOrMore, Group, 
                       stringEnd, Suppress, Literal, CaselessKeyword,
                       Optional, Forward, quotedString, Dict, nestedExpr)


# Used throughout as a variable/attr name.
var = Word(alphanums, "_" + alphanums)
dot_op = Suppress(Literal("."))

# MATCH STATEMENT
match = CaselessKeyword("MATCH")

graph = CaselessKeyword("GRAPH") | CaselessKeyword("SUBGRAPH")
graph.setParseAction(lambda t: t[0].lower())

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

# Recursive pattern match.
pattern = Forward() 
pattern << node.setResultsName("nodes", listAllMatches=True) + ZeroOrMore(
        edge.setResultsName("edges", listAllMatches=True) + pattern
)  

################ Transformations #######################

transformation = (
    CaselessKeyword("TRANSFER") |
    CaselessKeyword("PROJECT") |
    CaselessKeyword("COMBINE") 
)
transformation.setParseAction(lambda t: t[0].lower())


################### PREDICATE CLAUSES #######################

# Used for the creation of new nodes with combine or project.
new = CaselessKeyword("NEW")
new.setParseAction(lambda t: t[0].lower())

csv_pattern = Forward()
csv_pattern << var.setResultsName("pattern", listAllMatches=True) + ZeroOrMore(
     Suppress(Literal(",")) + csv_pattern
)

# Getter Setter Patterns.
left = new | var
attr = var + Literal(".") + var
right = attr("value_lookup") | quotedString("value")

gttr_sttr = (
    left("alias") + dot_op + var("key") + 
    Suppress(Literal("=")) + right
)

pred_pattern = Forward()
pred_pattern << gttr_sttr.setResultsName("pattern", listAllMatches=True) + ZeroOrMore(
    Suppress(Literal(",")) + pred_pattern
)

# DELETE
delete = CaselessKeyword("DELETE")
delete.setParseAction(lambda t: t[0].lower())
delete_clause = delete("predicate") + csv_pattern

# SET
setter = CaselessKeyword("SET")
setter.setParseAction(lambda t: t[0].lower())
set_clause = setter("predicate") + pred_pattern

# METHOD
method = CaselessKeyword("METHOD")

algorithm = (
    CaselessKeyword("ATTRS").setParseAction(lambda t: t[0].lower()) |
    CaselessKeyword("EDGES").setParseAction(lambda t: t[0].lower()) |
    CaselessKeyword("JACCARD").setParseAction(lambda t: t[0].lower())
)

#CaselessKeyword("OVER").setParseAction(lambda t: t[0].lower()) + 
#csv_pattern

method_clause = method("predicate") + algorithm("pattern")

predicate = (
    Optional(delete_clause("delete")) & Optional(set_clause("set")) & 
    Optional(method_clause("method"))
)


###################### QUERY ###########################
# Valid query clause.
clause = (
    # Something more like this. Create simpler dicts, not so
    # many groups.
    match("match") + Optional(graph("graph")) + pattern("match_pattern") +
    ZeroOrMore(
        transformation("transformation") + pattern("transform_pattern") +
        Optional(predicate),
    ).setResultsName("transformations", listAllMatches=True)
)

parser = OneOrMore(clause) + stringEnd

def parse_query(query):
    """
    """
    import ipdb; ipdb.set_trace()
    prsd = parser.parseString(query)
    match_pattern = prsd["match_pattern"]
    nodes = [{"node": node.asDict()} for node in match_pattern["nodes"]]
    edges = [{"edge": edge.asDict()} for edge in match_pattern["edges"]]
    transformations = prsd["transformations"]
    transformers = []
    for transformation in transformations:
        trans = transformation["transformation"]
        trans_pattern = transformation["transform_pattern"]
        trans_nodes = [{"node": node.asDict()} 
                       for node in trans_pattern["nodes"]]
        trans_edges = [{"edge": edge.asDict()}
                       for edge in trans_pattern["edges"]]
        method = transformation.get("method", "")
        if method:
            algorithm = method.asDict()["pattern"]
            if algorithm == "jaccard":
                pass
        else:
            algorithm = ""
        transformer = {
            trans: {
                "method": {algorithm: {}},
                "pattern": list(roundrobin(trans_nodes, trans_edges)),
                "set": [],
                "delete": {"alias": []}
            }
        }
        transformers.append(transformer)

    etl = {
        "extractor": {
            "networkx": {
                "type": prsd["graph"],
                "traversal": list(roundrobin(nodes, edges))
            }
        },
        "tansformers": transformers,
        "loader": {
            "networkx": {}
        }
    }
    return etl 


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
