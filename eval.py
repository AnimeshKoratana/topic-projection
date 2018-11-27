import numpy as np

def davies_bouldin(graph, clusters, centroids, n_clusters):
    # Calculate mean distance in each cluster
    mean_distances = dict((k, []) for k in range(n_clusters))
    for cluster, page in zip(clusters, graph):
        # Now we need to find the average distance to the other nodes in its cluster
        for j, dest_cluster in enumerate(clusters):
            dest_page = graph[j]
            if page != dest_page and dest_cluster == cluster:
                # We have a match! Find the distance from page to dest_page
                mean_distances[cluster].append(np.linalg.norm(page.embedding() - dest_page.embedding()))

    for k in mean_distances:
        if len(mean_distances[k]) != 0:
            mean_distances[k] = 1.0 * sum(mean_distances[k]) / len(mean_distances[k])
        else:
            mean_distances[k] = 0

    x = []
    for centroid_id in mean_distances:
        distances = []
        for comparison_id in mean_distances:
            if comparison_id != centroid_id:
                n = mean_distances[comparison_id] + mean_distances[centroid_id]
                d = np.linalg.norm(centroids[centroid_id] - centroids[comparison_id])
                z = n/d
                distances.append(z)
        x.append(max(distances))
    dist = sum(x) / len(x)
    return dist
