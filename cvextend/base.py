#!/usr/bin/env python3

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


def _get_object_fullname(o):
    module = o.__class__.__module__
    if module is None or module == str.__class__.__module__:
        return o.__class__.__name__
    else:
        return module + '.' + o.__class__.__name__


def process_grid_result(grid_result, step_names, **additional_info):
    grid_res = copy.deepcopy(grid_result)
    for key, value in additional_info.items():
        grid_res[key] = value

    # due to specifying steps in Pipeline as object instances,
    # results contain the instances themselves
    # instead return class name as string
    group_type_keys = []
    for group in step_names:
        type_group = 'type_' + group
        group_type_keys.append(type_group)
        param_group = 'param_' + group
        classes = grid_res[param_group]
        grid_res[type_group] = [_get_object_fullname(x) for x in classes]

    return grid_res
