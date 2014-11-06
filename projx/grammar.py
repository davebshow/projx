from pyparsing import (Word, alphas, alphanums, OneOrMore,
                       Group, stringEnd, Suppress, Literal)


word = Word(alphas)
verb = OneOrMore(word)
verb.setParseAction(lambda x: x[0].lower())
node = Word(
    '(' + alphanums + ')',
    '(' + alphanums + ':' + alphanums + ')'
)
edge = Suppress(Literal('-'))  # Right now these can be suppressed.
node.setParseAction(lambda x: x[0].lstrip('(').rstrip(')'))
pattern = node + OneOrMore(edge + node)
clause = Group(verb.setResultsName('verb') + pattern.setResultsName('pattern'))
grammar = OneOrMore(clause) + stringEnd
