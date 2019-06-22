#!/usr/bin/env python3

"""Basic functions for running nested cross-validation of sampling methods"""

# Authors: Lyubomir Danov <->
# License: -

import copy
from itertools import product as iter_product

import numpy
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import StratifiedKFold


def _expand_param_grid(steps, param_grid):
    param_grid_expanded = generate_param_grids(steps, param_grid)

    try:
        _ = ParameterGrid(param_grid_expanded)
    except Exception as e:
        raise e

    return param_grid_expanded


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


def _get_object_fullname(o):
    module = o.__class__.__module__
    if module is None or module == str.__class__.__module__:
        return o.__class__.__name__
    else:
        return module + '.' + o.__class__.__name__


def process_grid_result(grid_result, step_names, data_name):
    grid_res = copy.deepcopy(grid_result)
    grid_res['data_name'] = data_name

    full_step_names = ['param_' + str(x) for x in step_names]

    # due to specifying steps in Pipeline as object instances,
    # results contain the instances themselves
    # instead return class name as string
    for step in full_step_names:
        object_class = []
        for obj in grid_res[step]:
            object_class.append(_get_object_fullname(obj))
        object_class = numpy.array(object_class)
        grid_res[step] = object_class

    return grid_res


def generate_param_grids(steps, param_grids):

    final_params = []

    for estimator_names in iter_product(*steps.values()):
        current_grid = {}

        # Step_name and estimator_name should correspond
        # i.e preprocessor must be from pca and select.
        for step_name, estimator_name in zip(steps.keys(), estimator_names):
            for param, value in param_grids.get(estimator_name).items():
                if param == 'pipe_step_instance':
                    # Set actual estimator in pipeline
                    current_grid[step_name] = [value]
                else:
                    # Set parameters corresponding to above estimator
                    current_grid[step_name + '__' + param] = value
        # Append this dictionary to final params
        final_params.append(current_grid)

    return final_params
