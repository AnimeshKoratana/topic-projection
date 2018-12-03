import graph
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random

def chinese_whispers(g):
    G = g.graph
    g.add_path_weights()

    for page in G.nodes:
        page.cluster = page.idx


    print("Finished creating NetworkX graph")

    iterations = 10
    for z in range(0,iterations):
        gn = list(G.nodes)

        random.shuffle(gn)

        for node in gn:
            neighs = G.neighbors(node)
            classes = {}
            # do an inventory of the given nodes neighbours and edge weights
            for ne in neighs:
                if ne.cluster in classes:
                    classes[ne.cluster] += G[node][ne]['weight']
                else:
                    classes[ne.cluster] = G[node][ne]['weight']
            # find the class with the highest edge weight sum
            max = 0
            maxclass = 0
            for c in classes:
                if classes[c] > max:
                    max = classes[c]
                    maxclass = c
            # set the class of target node to the winning local class
            assert len(classes) > 0
            node.cluster = maxclass

    clusters = {}

    for page in G.nodes():
        if page.cluster not in clusters:
            clusters[page.cluster] = set()
        clusters[page.cluster].add(page)

    # Initialize color scheme so it spans the range of cluster values
    minCluster = None
    maxCluster = None
    for node in G.nodes():
        if minCluster is None or minCluster > node.cluster:
            minCluster = node.cluster
        if maxCluster is None or maxCluster < node.cluster:
            maxCluster = node.cluster
    rangeCluster = maxCluster - minCluster
    colors = [(node.cluster - minCluster)/rangeCluster for node in G.nodes()]

    # print(colors)

    fig = plt.gcf()
    fig.set_size_inches(10, 10)

    nx.draw_networkx(G, with_labels=False,cmap=plt.get_cmap('RdYlBu'), node_color=colors, font_color='white')

    plt.savefig("chinese_whispers.png")
    plt.show()
    return clusters

orig_graph = graph.Graph()
clusters = chinese_whispers(orig_graph)

print("total number of clusters:", len(clusters))
for cluster in clusters:
    print(cluster)
