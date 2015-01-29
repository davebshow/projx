# -*- coding: utf-8 -*-
from modules import loaders, neo4jxtrct, nxxtrct


class ETL(object):

    def __init__(self, etl):
        """
        A helper class that parses the ETL JSON, and initializes the
        required extractor, transformers, and loader. Also store key
        variables used in transformation/loading.

        :param etl: ETL JSON.
        """
        # Get the extractor info.
        try:
            self._extractor = etl["extractor"]
            self.extractor_name = self._extractor.keys()[0]
        except (KeyError, IndexError):
            raise Exception("Please define valid extractor")

        # Get the transformers.
        self.transformers = etl.get("transformers", [])

        # Get the loader info.
        try:
            self._loader = etl["loader"]
            self.loader_name = self._loader.keys()[0]
        except (KeyError, IndexError):
            raise Exception("Please define valid loader.")

        # Validate the extractor function.
        self._extractors = {}
        self._init_extractors()
        try:
            self.extractor = self.extractors[self.extractor_name]
        except KeyError:
            raise Exception(
                "{0} extractor not implemented".format(self.extractor_name)
            )

        # Validate the extractor function.
        self._streams = {}
        self._init_streams()
        try:
            self.stream = self.streams[self.extractor_name]
        except KeyError:
            raise Exception(
                "{0} stream not implemented".format(self.extractor_name)
            )

        # Validate the loader function.
        self._loaders = {}
        self._init_loaders()
        try:
            self.loader = self.loaders[self.loader_name]
        except KeyError:
            raise Exception(
                "{0} loader not implemented".format(self.loader_name)
            )

    def _get_extractors(self):
        return self._extractors
    extractors = property(fget=_get_extractors)

    def extractors_wrapper(self, extractor):
        """
        :param extractor: Str.
        """
        def wrapper(fn):
            self.extractors[extractor] = fn
        return wrapper

    def _init_extractors(self):
        """
        Update extractors dict to allow for extensible extractor
        functionality. Here we set the query attribute that will be passed to
        extractor match method. Also, each extractor function will return
        a projector class object that defines a match method.
        """
        @self.extractors_wrapper("networkx")
        def get_nx_extractor(graph):
            """
            :param graph: networkx.Graph
            :returns: projx.nx_extractor
            """
            return nxxtrct.nx_extractor(self._extractor[self.extractor_name], graph)

        @self.extractors_wrapper("neo4j")
        def get_neo4j_extractor(graph):
            """
            :returns: projx.nx_extractor
            """
            return neo4jxtrct.neo4j_extractor(self._extractor[self.extractor_name], graph)


    def _get_streams(self):
        return self._streams
    streams = property(fget=_get_streams)

    def streams_wrapper(self, stream):
        """
        :param stream: Str.
        """
        def wrapper(fn):
            self.streams[stream] = fn
        return wrapper

    def _init_streams(self):
        """
        Update streams dict to allow for extensible extractor
        functionality. Here we set the query attribute that will be passed to
        extractor match method. Also, each extractor function will return
        a projector class object that defines a match method.
        """
        @self.streams_wrapper("networkx")
        def get_nx_stream(extractor_context, graph):
            """
            :param graph: networkx.Graph
            :returns: projx.nx_extractor
            """
            return nxxtrct.nx_stream(extractor_context, graph)


    def _get_loader(self):
        return self._loaders
    loaders = property(fget=_get_loader)

    def loaders_wrapper(self, loader):
        def wrapper(fn):
            self.loaders[loader] = fn
        return wrapper

    def _init_loaders(self):
        """
        Update loaders dict to allow for extensible loader functionality.
        Returns a class/function that performs transformation and loads graph.
        """
        @self.loaders_wrapper("nx2nx")
        def get_nx2nx_loader(extractor, stream, transformers, graph):
            """
            :param tranformers: List of dicts.
            :extractor: function.
            :param graph: networkx.Graph
            :returns: projx.nx_loader
            """
            return loaders.nx2nx_loader(extractor, stream, transformers,
                                      self._loader[self.loader_name], graph)

        @self.loaders_wrapper("neo4j2nx")
        def get_neo4j2nx_loader(extractor, stream, transformers, graph):
            """
            :param tranformers: List of dicts.
            :extractor: function.
            :param graph: networkx.Graph
            :returns: projx.nx_loader
            """
            return loaders.neo4j2nx_loader(extractor, stream, transformers,
                                            self._loader[self.loader_name],
                                            graph)


        @self.loaders_wrapper("neo4j2edgelist")
        def get_neo4j2edgelist_loader(transformers, extractor, graph):
            """
            :param tranformers: List of dicts.
            :extractor: function.
            :param graph: networkx.Graph
            :returns: projx.nx_loader
            """
            return loaders.neo4j2edgelist_loader(transformers, extractor,
                                                  self._loader[self.loader_name],
                                                  graph)
