from pyparsing import (Word, alphanums, ZeroOrMore, OneOrMore,
                       Group, stringEnd, Suppress, Literal, Empty,
                       CaselessKeyword)
# ProjX verbs.
verb = (
    CaselessKeyword('MATCH') |
    CaselessKeyword('TRANSFER') |
    CaselessKeyword('TRANSFER_ATTRS') |
    CaselessKeyword('PROJECT')
)
verb.setParseAction(lambda x: x[0].lower())
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
    pattern.setResultsName('pattern')
)

# Tool for parsing.
parser = OneOrMore(clause) + stringEnd
