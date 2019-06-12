#!/usr/bin/env python3

"""Basic functions for running nested cross-validation of sampling methods"""

# Authors: Lyubomir Danov <->
# License: -

# from sklearn.model_selection import GridSearchCV, StratifiedKFold, ParameterGrid
# from itertools import product as iter_product
# import copy
# import numpy

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from .base import _get_object_fullname
import copy, pandas

class ExpandGrid(object):
    def __init__(self, gridcv, score_selection, step_names):
        if not isinstance(gridcv, (GridSearchCV, RandomizedSearchCV)):
            TypeError("grid is does not inherit from BaseSearchCV!")
        self.grid = gridcv
        self.score_selection = score_selection
        self.step_names = step_names
        self.scorers_best_params = None
        self.fitted_estimators = None
        self.final_result = None

    def refit_score(self, X_train, y_train, X_test, y_test, **kwargs):
        self.get_best_params()
        self.get_fitted_estimators(X_train, y_train)
        self.get_scores(X_test, y_test, **kwargs)
        if self.final_result is not None:
            return self.final_result


    def add_class_name(self, df, step_names):
        types_all_steps = []
        for step in step_names:
            type_step = 'type_' + step
            types_all_steps.append(type_step)
            param_step = 'param_' + step
            classes = df[param_step].values
            df[type_step] = [_get_object_fullname(x) for x in classes]
        return df, types_all_steps

    def get_best_params(self):
        '''
        Given a BaseSearchCV.cv_results_ object with results of all 
        parameter combinations, return a list of dictionaries containing
        the best hyperparameters for each combination of score and Pipeline step
        '''
        # TODO: replace pandas with numpy
        eval_df = copy.deepcopy(pandas.DataFrame(self.grid.cv_results_))
        eval_df, types_all_steps = self.add_class_name(eval_df, self.step_names)
        per_score = []

        for score_type in self.score_selection:

            score_key = score_type['score_search']
            selector = score_type['selector']

            # which columns to select
            retr_cols = types_all_steps + ['params']
            # for each unique value in each step from step_names
            # return those entries, where score_key is selector
            idx = eval_df.groupby(types_all_steps)[score_key].transform(selector)
            score_best_params = copy.deepcopy(
                eval_df.loc[idx == eval_df[score_key], retr_cols])

            # return score_name and scorer itself for ease of scoring
            score_best_params['score_name'] = score_type['score_name']
            score_best_params['scorer'] = score_type['scorer']

            per_score = per_score + score_best_params.to_dict('records')

        self.scorers_best_params = per_score
        return self

    def get_fitted_estimators(self, X_train, y_train):
        '''
        Given a estimator return a list of dictionaries containing
        fitted estimators for each score in the BaseSearchCV object
        '''
        fitted_estimators = copy.deepcopy(self.scorers_best_params)
        for best_param in fitted_estimators:
            cloned_estim = copy.deepcopy(self.grid.estimator)
            cloned_estim.set_params(**best_param['params'])
            cloned_estim.fit(X_train, y_train)
            best_param['estimator'] = cloned_estim

        self.fitted_estimators = fitted_estimators
        return self

    def get_scores(self, X_test, y_test, **kwargs):
        '''
        Given a BaseSearchCV.cv_results_ object with results of all 
        parameter combinations, return a list of dictionaries containing
        the best hyperparameters for each combination of score and Pipeline step
        '''
        # candidate_list
        final_result = copy.deepcopy(self.fitted_estimators)
        for estimator_dict in final_result:
            scorer = estimator_dict['scorer']
            estimator = estimator_dict['estimator']
            result = scorer(estimator, X_test, y_test)
            estimator_dict['score_value'] = result
            for key, value in kwargs.items():
                estimator_dict[key] = value
        self.final_result = final_result
        return self


    