import networkx as nx
import gensim.models as g
# model="data/enwiki_dbow/doc2vec.bin"
# m = g.Doc2Vec.load(model)

#inference hyper-parameters
start_alpha=0.01
infer_epoch=1000

class Page():
    def __init__(self, index, article_name, graph):
        assert(article_name is not None)

        self.idx = index
        self.cluster = None
        self._name = article_name
        self.graph = graph

        self._text = None
        self._title = None
        self._children = None
        self._embedding = None

        self.mean = None

    def __eq__(self, x):
        return self.idx == x.idx

    def __hash__(self):
        return self.idx

    def __sub__(self, other):
        assert(isinstance(other, Page))
        return self.graph.shortest_path_length(self, other)

    def __str__(self):
        return str(self.idx) + " " +  self.title()

    def shortest_path_length_to(self, other):
        return self.__sub__(other)

    def text(self):
        if self._text is None:
            with open('data/plaintext_articles/{}.txt'.format(self.title()), 'r') as t:
                text = t.read().split('\n')[1:]
                self._text = ' '.join(text).strip()
        return self._text

    def title(self):
        assert (self._name is not None)
        return self._name

    def children(self):
        return self._children

    def embedding(self):
        if self._embedding is not None:
            self._embedding = m.infer_vector(self.text().split(), alpha=start_alpha, steps=infer_epoch)
        return self._embedding

class Graph():
    def __init__(self, seed = None, num_pages= "all"):
        self.graph = nx.Graph()
        self.pg_map = {}
        print("Loading Dataset")
        l = open("data/wikispeedia_paths-and-graph/links.tsv", 'r')
        self._links = l.read().split("\n")[12:-1]
        l.close()

        self._articles = []
        a = open("data/wikispeedia_paths-and-graph/articles.tsv", 'r')
        lines = a.read().split("\n")[12:-1]
        for idx, line in enumerate(lines):
            article = line.strip(' ')
            article = article.strip("\t")
            article.strip()
            if article is not None:
                pg = Page(idx, article, self.graph)
                self._articles.append(pg)
                self.pg_map[article] = pg
        a.close()
        self._build_graph()
        print("Done.")

    def _build_graph(self):
        for page in self._articles:
            self.graph.add_node(page)

        for link in self._links:
            source, target = link.split('\t')
            source, target = source.strip(), target.strip()
            self.graph.add_edge(self.pg_map[source], self.pg_map[target], weight=0.1)
        x = []
        for page in self.graph.nodes:
            if len(list(self.graph.neighbors(page))) == 0:
                x.append(page)
        for p in x:
            self.graph.remove_node(p)

    def add_path_weights(self):
        l = open("data/wikispeedia_paths-and-graph/paths_finished.tsv", 'r')
        lines = l.read().split("\n")[16:-1]
        for idx, line in enumerate(lines):
            row = line.strip(' ')
            pathString = row.split("\t")[3].strip()
            path = pathString.split(";")
            pathPages = [self.pg_map[text] for text in path if text != '<']

            for i in range(len(pathPages)-1):
                decay = 1
                for j in range(i+1, len(pathPages)):
                    if self.graph.has_edge(pathPages[i], pathPages[j]):
                        old_weight = self.graph[pathPages[i]][pathPages[j]]['weight']
                        self.graph.remove_edge(pathPages[i], pathPages[j])
                    else:
                        old_weight = 0
                    self.graph.add_edge(pathPages[i], pathPages[j], weight = old_weight + (1/ 2**(decay)))
                    decay += 1
        l.close()

    def __len__(self):
        return len(list(self.graph.nodes))
    def nodes(self):
        return self._articles
