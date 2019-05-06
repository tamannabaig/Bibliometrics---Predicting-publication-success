#!/usr/bin/env python3
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np

if __name__ == "__main__":
    filename = 'data/database.pickle'
    with open(filename, 'rb') as data_source:
        papers = pickle.load(data_source)

    dataset = [p['abstract'] for p in papers.values()]
    titles = [p['title'] for p in papers.values()]

    # Clustering!
    vectorizer = TfidfVectorizer(max_df=0.5, max_features=10000,
                                 min_df=2, stop_words='english',
                                 use_idf=True)
    X = vectorizer.fit_transform(dataset)

    print("n_samples: %d, n_features: %d" % X.shape)
    print()

    # Determined experimentally
    exp_k = 27

    km = KMeans(n_clusters=exp_k, init='k-means++', max_iter=100, n_init=1,
                    verbose=True, random_state=5)

    print("Clustering sparse data with %s" % km)

    km.fit(X)
    print(km.random_state)
    groups = km.predict(X)

    order_centroids = km.cluster_centers_.argsort()[:, ::-1]
    terms = vectorizer.get_feature_names()

    with open('data/pp-fit.pickle', 'wb') as f:
        pickle.dump(groups, f)

    for g in range(min(groups),max(groups)+1):
        cluster = {}
        citations = []
        indices = np.argwhere(groups==g)

        # Print the relevant keywords and titles in the cluster
        print("Class: %d | Len: %d" % (g,indices.shape[0]))
        print("Terms:", end='')
        for t_idx in order_centroids[g,:10]:
            print(' %s' % terms[t_idx], end='')
        print()
        for idx in np.nditer(indices):
            print('\t' + titles[idx])

        # Add all the relevant papers to the cluster
        for idx in np.nditer(indices):
            k = list(papers.keys())[idx]
            if papers[k]['n_citation'] < 1000:
                cluster[k] = papers[k]
                citations.append(papers[k]['n_citation'])

        print("Median: " + str(np.median(citations)))

        with open('data/cluster-%d.pickle'%g, 'wb') as f:
            pickle.dump(cluster, f)
