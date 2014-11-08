from pyparsing import (Word, alphanums, ZeroOrMore, OneOrMore,
                       Group, stringEnd, Suppress, Literal, Empty,
                       CaselessKeyword, Optional)
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

# Node type pattern.
node_open = Suppress(Literal('('))
node_close = Suppress(Literal(')'))
node_content = (
    Word(alphanums, ':' + alphanums) |
    Empty().setParseAction(lambda t: ' ')
)
node = node_open + node_content + node_close

# Edge patterns.
edge = Suppress(Literal('-'))  # All edges are undirected, can be suppressed.

# Full pattern.
pattern = node + ZeroOrMore(edge + node)

# Valid query clause.
clause = Group(
    verb.setResultsName('verb') + 
    Optional(obj).setResultsName('object') +
    pattern.setResultsName('pattern')
)

# Parser.
parser = OneOrMore(clause) + stringEnd
