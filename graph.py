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
        a.close()
        self._build_graph()
        print("Done.")

    def _build_graph(self):
        def idx_of(text):
            for idx, a in enumerate(self._articles):
                if text == a.title():
                    return idx

        for page in self._articles:
            self.graph.add_node(page)

        for link in self._links:
            source, target = link.split('\t')
            source, target = source.strip(), target.strip()
            self.graph.add_edge(self._articles[idx_of(source)], self._articles[idx_of(target)], weight=1)
        x = []
        for page in self.graph.nodes:
            if len(list(self.graph.neighbors(page))) == 0:
                x.append(page)
        for p in x:
            self.graph.remove_node(p)

    def __len__(self):
        return len(list(self.graph.nodes))
    def nodes(self):
        return self._articles
