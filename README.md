# topic-projection
Projecting a Topic-Based Structure on Web Graphs

## Downloading the Datasets
You can access the Wikipedia dump file at the following location:
https://drive.google.com/drive/folders/0BwmD_VLjROrfbVNMYjFtRXItRjg

The wiki.hdf5 and the wiki_emb.hdf5 are the files used in this project.
(NOTE: the files used here require multiple gigabytes in storage)

You can access the Wikispeedia navigation path produced by the SNAP group at Stanford University here:
https://snap.stanford.edu/data/wikispeedia.html

The wikispeedia_paths-and-graph file was used to extract navigation paths

## Clustering
Extract clusters for LDA and chinese_whispers by running main.py

Extract clusters for K-means and MCL using the k_means.py and the mcl.py files, respectively.

## Topic Modeling
Topic model by using the LDA code on each of the clusters

##  Evaluation
Evaluate the clustering and topic modeling using the modularity and Davies Bouldin functions found in eval.py
