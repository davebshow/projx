# -*- coding: utf-8 -*-


class BasePatternElem(object):

    def __init__(self, elem_type, alias=None, gettr=(), 
                 settr=(), delete=False):
        self.type = elem_type
        self.alias = alias
        self.gettr = gettr
        self.settr = settr
        self.delete = delete


class Node(BasePatternElem):

    def __init__(self, elem_type, alias=None, gettr=(), 
                 settr=(), delete=False):
        super(Node, self).__init__(elem_type, alias, gettr, settr, delete)


class Rel(BasePatternElem):

    def __init__(self, elem_type, alias=None, gettr=(), 
                 settr=(), delete=False):
        super(Node, self).__init__(elem_type, alias, gettr, settr, delete)



class BaseOperation(object):

    def __init__(self, path=[], pattern=None, *args, **kwargs):        
        """
        :param path: An iterable of Node and/or Rel objects.
        :param pattern: projx.MatchPattern.

        """
        self.pattern = pattern
        if pattern:
            self.index = dict(pattern.node_alias())
            self.index.update(dict(pattern.edge_alias()))
        else:
            self.index = self.build_index(path)

    def build_index(self, path):
        index = {}
        for i, el in enumerate(path):
            if getattr(el, "alias", ""):
                index[el["alias"]] = i 
            elif getattr(el, "type", "") :
                index[el[self.type]] = i
            else:
                raise Error("Bad types")
        return index



class Match(BaseOperation):

    def __init__(self, *args, **kwargs):
        """
        :param pattern: Iterable of string type names
        for traversal criteria.

        """
        super(Match, self).__init__(*args, **kwargs)


class Transfer(BaseOperation):

    def __init__(self, ):
        """
        :param pattern: Iterable of string type names
        for traversal criteria.

        """
        super(Project, self).__init__(pattern)


class Project(BaseOperation):

    def __init__(self):
        """
        :param pattern: Iterable of string type names
        for traversal criteria.

        """
        super(Project, self).__init__(pattern)
