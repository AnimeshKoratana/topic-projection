import graph

import torch
import numpy as np
import eval

def vectorizedKMeans(graph, k):
    with torch.no_grad():
        centroids = torch.rand(k)
        old_centroids = centroids.clone() - 1
        i = 0
        while torch.norm(torch.add(centroids, -old_centroids)) > 1e-15:
            i+= 1
            print(i, torch.norm(torch.add(centroids, -old_centroids)))
            old_centroids = centroids.clone()
            for page in graph:
                closest_mean = None
                for mean in centroids:
                    if closest_mean is None or torch.norm(torch.add(page.embedding(), -mean)) < torch.norm(torch.add(page.embedding(), -closest_mean)):
                        closest_mean = mean
                page.mean = closest_mean

            for mean in centroids:
                sum_inputs = 0
                num = 0
                for page in graph:
                    if page.mean == mean:
                        sum_inputs += page.embedding()
                        num += 1
                mean = sum_inputs/num

def main():
    g = graph.Graph()
    vectorizedKMeans(g, 10)
    
if __name__ == '__main__':
    main()
