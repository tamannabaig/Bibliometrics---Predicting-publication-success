#!/usr/bin/env python3
import pickle
import sys
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
# Import Classifiers
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np

RAND_STATE = 5

if __name__ == "__main__":
    # Input clusters to process
    if not len(sys.argv) == 2:
        print("Usage: pp-transform <cluster-file>")
        sys.exit()

    class_txt = ''
    with open(sys.argv[1], 'r') as f:
        class_txt = f.read()

    total_tn = 0
    total_fp = 0
    total_fn = 0
    total_tp = 0
    # Iterate over clusters
    for c_str in class_txt.strip().split('\n'):
        # Load the cluster data
        c = int(c_str)
        print("Class %d" % c)
        with open('data/X-%d.pickle'%c, 'rb') as f:
            X = pickle.load(f)
        with open('data/X-text-%d.pickle'%c, 'rb') as f:
            X_text = pickle.load(f)
        with open('data/Y-%d.pickle'%c, 'rb') as f:
            Y = pickle.load(f)

        # Split numeric data
        X0_train, X0_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5, \
                random_state=RAND_STATE)

        # Standard scale
        scaler = StandardScaler()
        scaler.fit(X0_train)  #only scale w/ respect to training data
        X0_train = scaler.transform(X0_train)
        X0_test = scaler.transform(X0_test)
        X = scaler.transform(X) #for validation on whole group

        # Title TF-IDF
        count_vec = CountVectorizer()
        X1_count = count_vec.fit_transform([x[0] for x in X_text])
        tfidf_transformer = TfidfTransformer()
        X1_tfidf = tfidf_transformer.fit_transform(X1_count)
        X1_text_train, X1_text_test, Y_train, Y_test = train_test_split(X1_tfidf, \
                Y, test_size=0.5, random_state=RAND_STATE)

        # Abstract TF-IDF
        count_vec = CountVectorizer()
        X2_count = count_vec.fit_transform([x[1] for x in X_text])
        tfidf_transformer = TfidfTransformer()
        X2_tfidf = tfidf_transformer.fit_transform(X2_count)
        X2_text_train, X2_text_test, Y_train, Y_test = train_test_split(X2_tfidf, \
                Y, test_size=0.5, random_state=RAND_STATE)

        # Keyword TF-IDF
        count_vec = CountVectorizer()
        X3_count = count_vec.fit_transform([x[2] for x in X_text])
        tfidf_transformer = TfidfTransformer()
        X3_tfidf = tfidf_transformer.fit_transform(X3_count)
        X3_text_train, X3_text_test, Y_train, Y_test = train_test_split(X3_tfidf, \
                Y, test_size=0.5, random_state=RAND_STATE)

        # Random Forest Search Parameters
        param_grid = [
                {'n_estimators': [1,2,4,6,8,10,12],'max_depth': [None],'min_samples_split': [2], 'max_features':[2,4,6], 'random_state': [0]}
        ]
        param_grid1 = [
                {'n_estimators': [1,2,4,6,8,10,12],'max_depth': [None],'min_samples_split': [2], 'max_features':[1,2,3,4], 'random_state': [0]}
        ]

        # Optimize numeric Random Forest classifier
        clf0 = GridSearchCV(RandomForestClassifier(), param_grid, cv=5, scoring='accuracy')
        clf0.fit(X0_train, Y_train)

        print("Detailed classification report:")
        print(clf0.best_params_)
        print()
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()

        Y0_train_pred = clf0.predict(X0_train)
        Y_true, Y0_pred = Y_test, clf0.predict(X0_test)

        print(classification_report(Y_true, Y0_pred))
        print("ACCURACY: ", end='')
        print(accuracy_score(Y_true, Y0_pred))

        importances = clf0.best_estimator_.feature_importances_
        indices = np.argsort(importances)[::-1]
        for f in range(X.shape[1]):
            print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

        # Title MNB Classifier
        clf1 = MultinomialNB().fit(X1_text_train, Y_train)
        Y1_train_pred = clf1.predict(X1_text_train)
        Y_true, Y1_pred = Y_test, clf1.predict(X1_text_test)
        print(classification_report(Y_true, Y1_pred))
        print("ACCURACY: ", end='')
        print(accuracy_score(Y_true, Y1_pred))

        # Abstract MNB Classifier
        clf2 = MultinomialNB().fit(X2_text_train, Y_train)
        Y2_train_pred = clf2.predict(X2_text_train)
        Y_true, Y2_pred = Y_test, clf2.predict(X2_text_test)
        print(classification_report(Y_true, Y2_pred))
        print("ACCURACY: ", end='')
        print(accuracy_score(Y_true, Y2_pred))

        # Keywords MNB Classifier
        clf3 = MultinomialNB().fit(X3_text_train, Y_train)
        Y3_train_pred = clf3.predict(X3_text_train)
        Y_true, Y3_pred = Y_test, clf3.predict(X3_text_test)
        print(classification_report(Y_true, Y3_pred))
        print("ACCURACY: ", end='')
        print(accuracy_score(Y_true, Y3_pred))

        # Ensemble Classifier
        clf4 = GridSearchCV(RandomForestClassifier(), param_grid1, cv=5, scoring='accuracy')
        Y4_train = np.vstack((Y0_train_pred,Y1_train_pred,Y2_train_pred,Y3_train_pred)).T
        Y4_pred = np.vstack((Y0_pred,Y1_pred,Y2_pred,Y3_pred)).T
        clf4.fit(Y4_train, Y_train)

        print("Detailed classification report:")
        print(clf4.best_params_)
        print()
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()

        Y_true, Y_pred = Y_test, clf4.predict(Y4_pred)

        print(classification_report(Y_true, Y_pred))
        print("ACCURACY: ", end='')
        print(accuracy_score(Y_true, Y_pred))

        importances = clf4.best_estimator_.feature_importances_
        indices = np.argsort(importances)[::-1]
        for f in range(Y4_pred.shape[1]):
            print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

        # Accuracy calculation
        tn, fp, fn, tp = confusion_matrix(Y_true, Y_pred).ravel()
        total_tn += tn
        total_fp += fp
        total_fn += fn
        total_tp += tp

        print("OVERALL ACCURACY: ", end='')
        print((total_tp+total_tn)/(total_tp+total_fp+total_fn+total_tn))
