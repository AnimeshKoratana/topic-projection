import numpy as np
import graph
import whispering as w
from lda import LDA

import networkx as nx

def main():
    g = graph.Graph()
    clusters = w.chinese_whispers(g)

    # Now we need to run lda for each of the clusters that we developed
    topic_models = dict((k,LDA(clusters[k])) for k in clusters)
    for model in topic_models.values():
        model.fit(n_topics = 10, passes = 20, workers = 8)

    metagraph = nx.Graph()
    # now compare each cluster abs
    for c in clusters:
        metagraph.add_node(c)
    # Now do pairwise comparison to see how similar these topics are
    for ci in clusters.keys():
        for cj in clusters.keys():
            metagraph.add_edge(ci, cj, weight=topic_models[ci].compare(topic_models[cj]))





if __name__ == '__main__':
    main()
