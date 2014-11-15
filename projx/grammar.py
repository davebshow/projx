from pyparsing import (Word, alphanums, OneOrMore, ZeroOrMore, Group, 
                       stringEnd, Suppress, Literal, CaselessKeyword,
                       Optional, Forward, quotedString)


# Used throughout as a variable/attr name.
var = Word(alphanums, "_" + alphanums)
dot_op = Suppress(Literal("."))


################ VERBS AND DIRECT OBJECTS ###################
# ProjX verbs.
verb = (
    CaselessKeyword("MATCH") |
    CaselessKeyword("TRANSFER") |
    CaselessKeyword("PROJECT") |
    CaselessKeyword("COMBINE") |
    CaselessKeyword("RETURN")
)
verb.setParseAction(lambda t: t[0].lower())

# Objects of verbs.
obj = (
    CaselessKeyword("ATTRS") |
    CaselessKeyword("EDGES") |
    CaselessKeyword("GRAPH") |
    CaselessKeyword("SUBGRAPH") |
    CaselessKeyword("TABLE")
)

obj.setParseAction(lambda t: t[0].lower())


################### PREDICATE CLAUSES #######################
# Predicates
pred = (
    CaselessKeyword("DELETE") |
    CaselessKeyword("METHOD") |
    CaselessKeyword("WHERE") |
    CaselessKeyword("SET")
)
pred.setParseAction(lambda t: t[0].lower())

# Used for the creation of new nodes with combine.
new = CaselessKeyword("NEW")
new.setParseAction(lambda t: t[0].lower())

# Right part of getter setter.
right = (
    var.setResultsName("type2") + dot_op + var.setResultsName("attr2") |
    quotedString.setResultsName("attr2")
)

# This can be used with both SET and WHERE. 
gettr_settr = Group(
    var.setResultsName("type1") +
    dot_op +
    var.setResultsName("attr1") +
    Literal("=") +
    right
)

# Recursive definition for multiple predicate objects.
attr = gettr_settr | var
pred_obj = Forward()
pred_obj << attr.setResultsName("pred_objects", listAllMatches=True) + ZeroOrMore(
    Suppress(Literal(",")) +
    pred_obj
) 


################ NODE AND EDGE PATTERNS ###################
# Used for node and edge patterns.
seperator = Suppress(Literal(":"))
tp = seperator + Word(alphanums, "_" + alphanums)

# Node type pattern.
node_open = Suppress(Literal("("))
node_close = Suppress(Literal(")"))
node_content = Group(
    var.setResultsName("alias") +
    Optional(tp).setResultsName("type")
)

node = node_open + node_content + node_close

# Edge patterns.
edge_marker = Suppress(Literal("-"))
edge_open = Suppress(Literal("["))
edge_close = Suppress(Literal("]"))
edge_content = edge_open + Group(
    var.setResultsName("alias") +
    Optional(tp).setResultsName("type")
) + edge_close

edge = edge_marker + Optional(edge_content + edge_marker)

# Recursive pattern match.
pattern = Forward() 
pattern << node.setResultsName("nodes", listAllMatches=True) + ZeroOrMore(
    edge.setResultsName("edges", listAllMatches=True) + 
    pattern
)


###################### QUERY ###########################
# Valid query clause.
clause = Group(
    verb.setResultsName("verb") + 
    Optional(obj).setResultsName("object") +
    pattern.setResultsName("pattern") +
    Optional(pred.setResultsName("predicate") +
             pred_obj.setResultsName("pred_obj")).setResultsName("predicate")
)

parser = OneOrMore(clause) + stringEnd
