#!/usr/bin/env python3

"""Utility functions for running nested cross-validation of sampling methods"""

# Authors: Lyubomir Danov <->
# License: -

import pytest

# from ..cv_utils import generate_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_breast_cancer


# def test_generate_pipeline_clf():

#     X, y = load_breast_cancer(return_X_y=True)

#     search_space = [
#         {
#             'clf': [RandomForestClassifier()],
#             'clf__n_estimators': [15],
#             'clf__random_state': [0]
#         },
#     ]
#     disable_cv = [(slice(None), slice(None))]
#     test_grid = GridSearchCV(
#         estimator = generate_pipeline(),
#         param_grid = search_space,
#         cv = disable_cv,
#         scoring = 'accuracy'
#     )
#     test_rf = RandomForestClassifier(n_estimators=15, random_state=0)
#     test_rf.fit(X, y)
#     test_grid.fit(X, y)

#     preds_rf = test_rf.predict(X)
#     preds_grid = test_grid.predict(X)
#     pred_proba_rf = test_rf.predict_proba(X)
#     pred_proba_grid = test_grid.predict_proba(X)
#     assert (preds_grid == preds_rf).all()
#     assert (pred_proba_grid == pred_proba_rf).all()


# def test_find_model_params():

#     X, y = load_breast_cancer(return_X_y=True)

#     search_space = [
#         {
#             'clf': [RandomForestClassifier()],
#             'clf__n_estimators': [15],
#             'clf__random_state': [0]
#         },
#     ]
#     disable_cv = [(slice(None), slice(None))]
#     test_grid = GridSearchCV(
#         estimator = generate_pipeline(),
#         param_grid = search_space,
#         cv = disable_cv,
#         scoring = 'accuracy'
#     )
#     test_rf = RandomForestClassifier(n_estimators=15, random_state=0)
#     test_rf.fit(X, y)
#     test_grid.fit(X, y)

#     preds_rf = test_rf.predict(X)
#     preds_grid = test_grid.predict(X)
#     pred_proba_rf = test_rf.predict_proba(X)
#     pred_proba_grid = test_grid.predict_proba(X)
#     assert (preds_grid == preds_rf).all()
#     assert (pred_proba_grid == pred_proba_rf).all()