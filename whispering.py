import graph
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random

orig_graph = graph.Graph()

G = orig_graph.graph
# G = nx.Graph()
#
for page in G.nodes:
    # G.add_node(page.idx)
    G.node[page]['class'] = page.idx + 1
#
# maxValue = None
#
# for page in orig_graph:
#     for page2 in orig_graph.subgraph:
#         if maxValue is None or maxValue < np.linalg.norm(np.subtract(page.embedding(), page2.embedding())):
#             maxValue = np.linalg.norm(np.subtract(page.embedding(), page2.embedding()))
#
# for page in orig_graph:
#     for page2 in orig_graph.subgraph:
#         print(np.subtract(page.embedding(), page2.embedding()))
#         if page != page2:
#             distance = 1-(np.linalg.norm(np.subtract(page.embedding(),page2.embedding())))/maxValue
#             print(distance)
#             G.add_edge(page.idx, page2.idx, weight=distance)
#
# used = set()
# for page in orig_graph:
#     used.add(page.idx)
#     for page2 in orig_graph.subgraph:
#         if page2.idx not in used:
#             cosSim = np.dot(page.embedding(), page2.embedding())/(np.linalg.norm(page.embedding()) * np.linalg.norm(page2.embedding()))
#             print(cosSim)
#             G.add_edge(page.idx, page2.idx, weight=cosSim)


print("Finished creating NetworkX graph")

iterations = 10
for z in range(0,iterations):
    gn = list(G.nodes)
    # print(gn)
    # gn = [int(gn[i]) for i in range(len(gn))]
    # I randomize the nodes to give me an arbitrary start point
    random.shuffle(gn)
    # print(gn)
    # i=0
    for node in gn:
        neighs = G[node]
        print(neighs)
        classes = {}
        # do an inventory of the given nodes neighbours and edge weights
        for ne in neighs:
            if isinstance(ne, int) :
                # print(classes)
                # if z == 0 and i == 0:
                #     print(G.node[ne]['class'])
                if G.node[ne]['class'] in classes:
                    classes[G.node[ne]['class']] += G[node][ne]['weight']
                else:
                    classes[G.node[ne]['class']] = G[node][ne]['weight']
        # find the class with the highest edge weight sum
        max = 0
        maxclass = 0
        print(classes)
        for c in classes:
            if classes[c] > max:
                max = classes[c]
                maxclass = c
        # set the class of target node to the winning local class
        G.node[node]['class'] = maxclass

        # i+=1

for node in G.nodes():
    print(G.node[node]['class'])

colors = [1. / G.node[node]['class'] for node in G.nodes()]

print(colors)

fig = plt.gcf()
fig.set_size_inches(10, 10)

nx.draw_networkx(G, cmap=plt.get_cmap('jet'), node_color=colors, font_color='white')

plt.savefig('whispering.png')
