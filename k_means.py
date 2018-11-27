import graph

import torch
import numpy as np

def vectorizedKMeans(graph, k):
    centroids = torch.rand(k)
    old_centroids = centroids.copy() - 1

    while torch.norm(torch.add(centroids, -old_centroids)) < 1e-15:
        old_centroids = centroids.copy()
        for page in graph:
            closest_mean = None
            for mean in centroids:
                if closest_mean == None or torch.norm(torch.add(page.embedding(), -mean)) < torch.norm(torch.add(page.embedding(), -closest_mean)):
                    closest_mean = mean
            page.mean = closest_mean

        for mean in centroids:
            sum_inputs = 0
            num = 0
            for page in graph:
                if page.mean = mean:
                    sum_inputs += page.embedding()
                    num += 1
            mean = sum_inputs/num

g = graph.Graph()

vectorizedKMeans(g, 100000)
