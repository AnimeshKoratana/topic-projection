import graph
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random

def k_means(g, k):
    G = g.graph
    g.add_path_weights()

    centroid_indices = np.random.choice(len(G.nodes), k, replace=False)
    nodes = list(G.nodes)
    centroids = set([nodes[i] for i in centroid_indices])

    clusterAssignments = {}
    for page in G.nodes:
        clusterAssignments[page] = centroids[np.random.randint(k)]

    # clusterAssignments = {(i, set()) for i in centroids}
    # for page in G.nodes:
    #     clusterAssignments[np.random.randint(k)] = page

    old_centroids = None

    while old_centroids != centroids:
        old_centroids = centroids.copy()

        clusterSets = {(i, set()) for i in centroids}

        for page in G.nodes:
            closestDistance = 0
            closest = None
            for centroid in centroids:
                currDistance = max(centroid-page, page-centroid)
                if closest is None or currDistance > closest:
                    closest = centroid
                    closestDistance = currDistance

            clusterAssignments[page] = closest
            clusterSets[closest].add(page)

        centroids = set()

        for centroid in old_centroids:
            assignedNodes = clusterSets[centroid]
            centralNode = None
            centralDistance = 0
            for node in assignedNodes:
                distance = 0
                for node2 in assignedNodes.copy():
                    distance += (node-node2)
                if centralNode is None or distance > centralDistance:
                    centralNode = node
                    centralDistance = distance
            centroids.add()




    for cluster in clusters:
        for page in clusters[cluster]:
            page.cluster = cluster
