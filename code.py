#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 14:41:45 2017

@author: micheles
"""

import numpy as np
import matplotlib.pyplot as plt 
import sklearn
import time

# Load database
import scipy.io as sio
database = sio.loadmat('./data/mnist-original.mat')

all_data = database['data'].T
GT = database['label'].T

# What we have?
n_samples, m_features = all_data.shape
print 'Shape: ' + str(all_data.shape)
print 'GT values: ' + str(np.unique(GT))

one_sample = np.reshape(all_data[12,:],(int(np.sqrt(m_features)),int(np.sqrt(m_features))))
plt.imshow(one_sample)
print 'sample range is: MAX=%d, min=%d' % (np.max(one_sample),np.min(one_sample))
how_many_samples_per_class = plt.hist(GT)
print 'How many samples per class: ' + str(how_many_samples_per_class[0])

# Split data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
        all_data, GT, test_size=0.50, random_state=42)

X_test, X_valid, y_test, y_valid = train_test_split(
        X_test, y_test, test_size=0.50, random_state=42)

y_train = y_train.reshape(-1)
y_valid = y_valid.reshape(-1)
y_test = y_test.reshape(-1)

print 'Shape (train): ' + str(X_train.shape)
print 'Shape (valid): ' + str(X_valid.shape)
print 'Shape (test): ' + str(X_test.shape)


# Fast SVM trial
from sklearn.svm import LinearSVC
clf = LinearSVC()

t = time.time()
print 'Training linear SVM...'
clf.fit(X_train,y_train)
print 'Done in (sec): %.3f' % (time.time() - t)

y_valid_pred = clf.predict(X_valid)
from sklearn.metrics import classification_report
print(classification_report(y_valid, y_valid_pred, target_names=['Class_' + str(i) for i in np.unique(GT)]))
print 'Total accuracy (%%): %.3f' % (clf.score(X_valid,y_valid)*100)

# Let's scale the data first
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_valid = scaler.transform(X_valid)
X_test = scaler.transform(X_test)

print 'Shape (train): ' + str(X_train.shape)
print 'Shape (valid): ' + str(X_valid.shape)
print 'Shape (test): ' + str(X_test.shape)


# SVM trial after scaling
t = time.time()
print 'Training linear SVM...'
clf.fit(X_train,y_train)
print 'Done in (sec): %.3f' % (time.time() - t)

y_valid_pred = clf.predict(X_valid)
from sklearn.metrics import classification_report
print(classification_report(y_valid, y_valid_pred, target_names=['Class_' + str(i) for i in np.unique(GT)]))
print 'Total accuracy (%%): %.3f' % (clf.score(X_valid,y_valid)*100)



from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(X_train, y_train)

y_valid_pred = clf.predict(X_valid)
print(classification_report(y_valid, y_valid_pred, target_names=['Class_' + str(i) for i in np.unique(GT)]))
print 'Total accuracy (%%): %.3f' % (clf.score(X_valid,y_valid)*100)

from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

opt = BayesSearchCV(SVC(), { 'C': Real(1e-6, 1e+6, prior='log-uniform'),
                    'gamma': Real(1e-6, 1e+1, prior='log-uniform'),
                    'degree': Integer(1,8), 'kernel': Categorical(['linear', 'poly', 'rbf']), },
                    n_iter=32, n_jobs=-1)
t = time.time()
opt.fit(X_train, y_train)
print 'Optimisation of SVM done in (sec): %.3f' % (time.time() - t)
#print(opt.score(X_test, y_test))


# The show begin
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB


from skopt.space import Real, Categorical, Integer
from skopt import BayesSearchCV                             # !!! It needs SKOPT 0.4

# used to try different model classes
pipe = Pipeline([
    ('model', SVC())
])

# single categorical value of 'model' parameter is used  to set the model class
lin_search = {
    'model': Categorical([LinearSVC()]),
    'model__C': Real(1e-6, 1e+6, prior='log-uniform'),
}

knn_search = {
    'model': Categorical([KNeighborsClassifier()]),
    'model__n_neighbors': Integer(1, 10),
}

rfc_search = {
    'model': Categorical([RandomForestClassifier()]),
    'model__max_depth': Integer(1, 10),
    'model__n_estimators': Integer(1, 20),
    'model__max_features': Integer(1, 5),
}

mlp_search = {
    'model': Categorical([MLPClassifier()]),
    'model__alpha': Integer(1, 10),
}

nb_search = {
    'model': Categorical([GaussianNB()])
}

ab_search = {
    'model': Categorical([AdaBoostClassifier()])
}

dtc_search = {
    'model': Categorical([DecisionTreeClassifier()]),
    'model__max_depth': Integer(1, 32),
    'model__min_samples_split': Real(1e-3, 1.0, prior='log-uniform'),
}

svc_search = {
    'model': Categorical([SVC()]),
    'model__C': Real(1e-6, 1e+6, prior='log-uniform'),
    'model__gamma': Real(1e-6, 1e+1, prior='log-uniform'),
    'model__degree': Integer(1, 8),
    'model__kernel': Categorical(['linear', 'poly', 'rbf']),
}

opt = BayesSearchCV(pipe,
    [(lin_search, 16), (dtc_search, 24), (svc_search, 64),
     (nb_search, 1), (ab_search, 1), (knn_search, 16),
     (rfc_search, 32), (mlp_search, 16)],  # (parameter space, # of evaluations)
                    n_jobs=20)

opt.fit(X_train, GT_train)















