import graph
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from chinese_whispers import chinese_whispers

orig_graph = graph.Graph()

G = nx.Graph()

for page in orig_graph:
    G.add_node(page.idx)

i = 1
j = 1
for page in orig_graph:
    j = 1
    for page2 in orig_graph.subgraph:
        print("Epoch", i, "Iteration", j)
        # print(page.embedding(), page2.embedding())
        distance = np.absolute(np.linalg.norm(page.embedding()) - np.linalg.norm(page2.embedding()))
        G.add_edge(page.idx, page2.idx, weight=distance)
        j += 1
    i += 1

print("Finished creating NetworkX graph")

chinese_whispers(G, weighting='top')

colors = [1. / G.node[node]['label'] for node in G.nodes()]

fig = plt.gcf()
fig.set_size_inches(10, 10)

nx.draw_networkx(G, cmap=plt.get_cmap('jet'), node_color=colors, font_color='white')

plt.show()
