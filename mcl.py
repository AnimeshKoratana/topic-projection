import markov_clustering as mc
import networkx as nx
import random
from matplotlib.pylab import cm, axis, savefig
import eval
import graph

class MCL():
    def __init__(self, g):
        g.add_path_weights()
        self.g = g
        self.graph = g.graph
        self.matrix = nx.to_scipy_sparse_matrix(self.graph)
        self.result = None
        self.clusters = None

    def fit(self, inflation = 2.0):
        if self.result is None:
            self.result = mc.run_mcl(self.matrix, inflation=inflation)
            self.clusters = mc.get_clusters(self.result)
            print(self.clusters[0][0])
            print("number of clusters:", len(self.clusters))
        return self.clusters

    def draw_graph(self, **kwargs):
        # map node to cluster id for colors
        cluster_map = {node: i for i, cluster in enumerate(self.clusters) for node in cluster}
        colors = [cluster_map[i] for i in range(len(self.graph.nodes()))]

        # if colormap not specified in kwargs, use a default
        if not kwargs.get("cmap", False):
            kwargs["cmap"] = cm.tab20

        # draw
        nx.draw_networkx(self.graph, node_color=colors, **kwargs)
        axis("off")
        savefig("mcl.png")

    def cluster_dict(self):
        assert(self.clusters)
        x = dict((i, []) for i in range(len(self.clusters)))
        for c in range(len(self.clusters)):
            for i in c:
                x[c].append(self.g._articles[i])
        return x


def main():
    g = graph.Graph()
    l = MCL(g)
    l.fit()
    print(eval.modularity(l.matrix, l.clusters))
    # l.draw_graph(node_size=50, with_labels=False, edge_color="silver")


if __name__ == '__main__':
    main()
