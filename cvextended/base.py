#!/usr/bin/env python3

"""Basic functions for running nested cross-validation of sampling methods"""

# Authors: Lyubomir Danov <->
# License: -

from sklearn.model_selection import GridSearchCV, StratifiedKFold, ParameterGrid
from itertools import product as iter_product
import copy
import numpy


def _expand_param_grid(steps, param_grid):
    param_grid_expanded = generate_param_grids(steps, param_grid)
    step_names = list(steps.keys())

    try:
        _ = ParameterGrid(param_grid_expanded)
    except Exception as e:
        raise e

    return param_grid_expanded, step_names


def _get_object_fullname(o):
    module = o.__class__.__module__
    if module is None or module == str.__class__.__module__:
        return o.__class__.__name__
    else:
        return module + '.' + o.__class__.__name__


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


# def get_model_scores(data_name: str, X, y,
#                      grid, step_names):

#     grid.fit(X, y)
#     grid_result_df = process_grid_result(grid.cv_results_,
#                                          step_names=step_names,
#                                          data_name=data_name)

#     return grid_result_df


def generate_param_grids(steps, param_grids):

    final_params = []
    # step_keys, step_values = steps.items()

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

def transform_score_selection(score_selection):
    sklearn_score_dict = {}
    for score in score_selection:
        sklearn_score_dict[score['score_name']] = score['scorer']
    
    return sklearn_score_dict

def add_class_name(df, step_names):
    types_all_steps = []
    for step in step_names:
        type_step = 'type_' + step
        types_all_steps.append(type_step)
        param_step = 'param_' + step
        classes = df[param_step].values
        df[type_step] = [_get_object_fullname(x) for x in classes]
    return df, types_all_steps