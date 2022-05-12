# ------------------------------------------------------------------------------
# Copyright (c) RocketML
# ------------------------------------------------------------------------------

#!/usr/bin/env python
# coding: utf-8

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
import sys
import os
import requests

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import logging
import subprocess

logger = logging.getLogger('RMLDNN|')
handler = logging.StreamHandler()
log_format = logging.Formatter(
    'Rank 0: %(asctime)s:%(name)s:%(levelname)s:%(message)s')
handler.setFormatter(log_format)
logger.addHandler(handler)

# def get_mlflow_tracking_url():
#     from requests import get
#     ip = get('https://api.ipify.org').text
#     url = "https://w"+''.join(ip.split('.'))+'.babyrocket.net/mlflow'
#     return url


def check_or_restart_mlflow():
    tracking_uri = os.environ.get("TRACKING_URL")
    experiments_url = tracking_uri + "/api/2.0/mlflow/experiments/list"
    try:
        requests.get(experiments_url, timeout=2.0)
        logger.debug("MLflow connection is successful")
    except:
        logger.debug("Restarting MLflow...")
        subprocess.check_output("sudo service mlflow restart".split())
        logger.debug("MLflow connection is successful")
        requests.get(experiments_url, timeout=2.0)


def create_experiment_runs():
    categories = ['alt.atheism', 'soc.religion.christian',
                  'comp.graphics', 'sci.med']
    mlflow_params = {}
    mlflow_metrics = {}

    # In[74]:

    tracking_uri = os.environ.get("TRACKING_URL")
    client = MlflowClient(tracking_uri=tracking_uri)
    mlflow.set_tracking_uri(tracking_uri)
    experiments = client.list_experiments()
    experiment_names = []
    for exp in experiments:
        experiment_names.append(exp.name)
    experiment_name = "nlp_demo_2"
    if experiment_name not in experiment_names:
        mlflow.create_experiment(experiment_name)
    mlflow.set_experiment(experiment_name)

    # In[75]:

    twenty_train = fetch_20newsgroups(subset='train',
                                      categories=categories, shuffle=True, random_state=42)

    # In[76]:

    twenty_train.target_names

    # In[77]:

    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(twenty_train.data)
    mlflow_params["samples"] = X_train_counts.shape[0]
    mlflow_params["features"] = X_train_counts.shape[1]

    # In[78]:

    use_idf = True
    tf_transformer = TfidfTransformer(use_idf=use_idf).fit(X_train_counts)
    X_train_tf = tf_transformer.transform(X_train_counts)
    X_train_tf.shape
    mlflow_params["use_idf"] = use_idf

    # In[79]:

    from sklearn.pipeline import Pipeline
    text_clf = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', SGDClassifier(loss='hinge', penalty='l2',
                              alpha=1e-3, random_state=42,
                              max_iter=5, tol=None))
    ])

    mlflow_params["classifier"] = text_clf.steps[2][1].__class__.__name__

    # In[80]:

    text_clf.fit(twenty_train.data, twenty_train.target)

    # In[81]:

    import numpy as np
    twenty_test = fetch_20newsgroups(subset='test',
                                     categories=categories, shuffle=True, random_state=42)
    docs_test = twenty_test.data
    predicted = text_clf.predict(docs_test)
    accuracy = np.mean(predicted == twenty_test.target)
    mlflow_metrics["accuracy"] = accuracy

    # In[82]:

    from sklearn import metrics

    # In[83]:

    report = metrics.classification_report(twenty_test.target, predicted,
                                           target_names=twenty_test.target_names)

    # In[84]:

    fp = open("report.txt", "w")
    fp.write(report)
    fp.close()

    # In[85]:

    with mlflow.start_run():
        mlflow.log_params(mlflow_params)
        mlflow.log_metrics(mlflow_metrics)
        mlflow.sklearn.log_model(text_clf, "model")
        mlflow.log_artifact("report.txt")


def main():
    check_or_restart_mlflow()
    create_experiment_runs()
    print('creating experiment with runs: DONE')


if __name__ == '__main__':
    try:
        main()
    except Exception as ex:
        exceptionMessage = str(ex)
        print(exceptionMessage)
        print('created experiment with runs: FAILED')
