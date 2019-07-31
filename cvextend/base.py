"""Basic functions for running nested cross-validation of sampling methods"""

# Authors: Lyubomir Danov <->
# License: -

import copy

import numpy
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold


def get_grid(estimator, param_grid, scoring, n_splits,
             random_state, refit=False, verbose=1):

    sfk_cv = StratifiedKFold(
        n_splits=n_splits, shuffle=True,
        random_state=random_state)

    grid = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        cv=sfk_cv,
        scoring=scoring,
        return_train_score=True,
        iid=False,
        refit=refit,
        verbose=verbose
    )
    return grid


def get_cv_grid(estimator, param_grid, scoring,
                cv=StratifiedKFold(shuffle=True),
                refit=False, verbose=1):

    grid = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        return_train_score=True,
        iid=False,
        refit=refit,
        verbose=verbose
    )
    return grid
