from pyparsing import (Word, alphanums, OneOrMore, ZeroOrMore, Group, 
                       stringEnd, Suppress, Literal, CaselessKeyword,
                       Optional, Forward)
# ProjX verbs.
verb = (
    CaselessKeyword('MATCH') |
    CaselessKeyword('TRANSFER') |
    CaselessKeyword('PROJECT')
)
verb.setParseAction(lambda t: t[0].lower())

# Objects of verbs.
obj = (CaselessKeyword('ATTRS') | CaselessKeyword('EDGES'))
obj.setParseAction(lambda t: t[0].lower())

seperator = Suppress(Literal(':'))

# Node type pattern.
node_open = Suppress(Literal('('))
node_close = Suppress(Literal(')'))
node_alias = Word(alphanums, '_' + alphanums)
node_type = seperator + Word(alphanums, '_' + alphanums)
node_content = Group(
    node_alias.setResultsName('alias') +
    Optional(node_type).setResultsName('type')
)

node = node_open + node_content + node_close

# Edge patterns.
edge_marker = Suppress(Literal('-'))
edge_open = Suppress(Literal('['))
edge_close = Suppress(Literal(']'))
edge_alias = Word(alphanums, '_' + alphanums)
edge_type = seperator + Word(alphanums, '_' + alphanums)
edge_content = edge_open + Group(
    edge_alias.setResultsName('alias') +
    Optional(edge_type).setResultsName('type')
) + edge_close

edge = edge_marker + Optional(edge_content + edge_marker)

# Recursive pattern match.
pattern = Forward() 
pattern << node.setResultsName('nodes', listAllMatches=True) + ZeroOrMore(
    edge.setResultsName('edges', listAllMatches=True) + 
    pattern
)

# Valid query clause.
clause = Group(
    verb.setResultsName('verb') + 
    Optional(obj).setResultsName('object') +
    pattern.setResultsName('pattern')
)

parser = OneOrMore(clause) + stringEnd
