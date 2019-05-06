#!/usr/bin/env python3
import pickle
from sklearn.cluster import KMeans
import numpy as np
import sys

if __name__ == "__main__":
    if not len(sys.argv) == 2:
        print("Usage: pp-transform <cluster-max>")
        sys.exit()

    c_max = int(sys.argv[1])

    # Iterate over all clusters to process
    for c in range(0,c_max+1):
        print("Cluster: %d" % c)
        # Load the papers in that cluster
        filename = 'data/cluster-%d.pickle' % c
        with open(filename, 'rb') as f:
            papers = pickle.load(f)

        # Calculate the median citations in each cluster
        cite_list = [p['n_citation'] for p in papers.values()]
        median = np.median(cite_list)

        # Features: title_count, abstract_count, keyword_count, year, author_count, page_count
        X = np.empty([0, 6],dtype=int)
        # Features: Title, Abstract, Keywords
        X_text = []
        # Target
        Y = np.empty(0,dtype=int)
        for idx,p in enumerate(papers.values()):
            if not p['page_count'] == None:
                # Extracting the numerica data
                data = []
                data.append(len(p['title'].split(' ')))
                data.append(len(p['abstract'].split(' ')))
                data.append(len(p['keywords']))
                data.append(p['year'])
                data.append(len(p['authors']))
                data.append(p['page_count'])
                X = np.append(X,[data],axis=0)
                # Extracting the text data
                X_text.append([p['title'], p['abstract'], ' '.join(p['keywords'])])
                # Generating the ground truth
                if p['n_citation'] < median:
                    Y = np.append(Y,0)
                else:
                    Y = np.append(Y,1)

        with open('data/X-%d.pickle'%c, 'wb') as f:
            pickle.dump(X, f)
        with open('data/X-text-%d.pickle'%c, 'wb') as f:
            pickle.dump(X_text, f)
        with open('data/Y-%d.pickle'%c, 'wb') as f:
            pickle.dump(Y, f)
