import graph
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random

orig_graph = graph.Graph()

G = orig_graph.graph

for page in G.nodes:
    page.cluster = page.idx + 1


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
        # if len(classes) > 0:
        #     node.cluster = maxclass

colors = [1. / node.cluster for node in G.nodes()]

# print(colors)

fig = plt.gcf()
fig.set_size_inches(10, 10)

nx.draw_networkx(G, with_labels=False,cmap=plt.get_cmap('RdYlBu'), node_color=colors, font_color='white')

plt.show()
