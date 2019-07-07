#!/usr/bin/env python3

"""NestedEvaluationGrid class for running nested cross-validation of sampling methods"""

# Authors: Lyubomir Danov <->
# License: -

import copy

import pandas

from .score_grid import ScoreGrid


class NestedEvaluationGrid(object):
    def __init__(self, gridcv, score_grid, group_names):

        if not isinstance(score_grid, ScoreGrid):
            TypeError("score_grid is does not inherit from ScoreGrid!")

        self.cv_results = copy.deepcopy(gridcv.cv_results_)
        self.cv_estimator = copy.deepcopy(gridcv.estimator)
        self.score_grid = score_grid
        self.groups = group_names
        self.scorers_best_params = None
        self.fitted_estimators = None
        self.final_result = None

    def refit_score(self, X_train, y_train, X_test, y_test, **kwargs):
        self.add_metainfo_results()
        self.get_best_params()
        self.get_fitted_estimators(X_train, y_train)
        self.get_scores(X_test, y_test, **kwargs)
        if self.final_result is not None:
            return self.final_result

    def add_metainfo_results(self):
        eval_df = copy.deepcopy(pandas.DataFrame(self.cv_results))
        eval_df, group_type_keys = self.process_result(eval_df, self.groups)

        self.eval_df = eval_df
        self.group_type_keys = group_type_keys
        return self

    def get_best_params(self):
        '''
        Given a BaseSearchCV.cv_results_ object with results of all 
        parameter combinations, return a list of dictionaries containing
        the best hyperparameters for each combination of score and Pipeline step
        '''
        # TODO: replace pandas with numpy
        eval_df = self.eval_df
        per_score = []

        for score_type in self.score_grid.score_selection:

            score_key = score_type['score_key']
            score_criteria = score_type['score_criteria']

            # which columns to select
            retr_cols = self.group_type_keys + ['params']
            # for each unique value in each group from groups
            # return entries where score_key corresponds to score_criteria
            idx = eval_df.groupby(self.group_type_keys)[
                score_key].transform(score_criteria)
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
            cloned_estim = copy.deepcopy(self.cv_estimator)
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

    @staticmethod
    def process_result(result, step_names, **additional_info):
        for key, value in additional_info.items():
            result[key] = value

        # due to specifying steps in Pipeline as object instances,
        # results contain the instances themselves
        # instead return class name as string
        group_type_keys = []
        for group in step_names:
            type_group = 'type_' + group
            group_type_keys.append(type_group)
            param_group = 'param_' + group
            classes = result[param_group]
            result[type_group] = [
                NestedEvaluationGrid._get_object_fullname(x)
                for x in classes
            ]

        return result, group_type_keys

    @staticmethod
    def _get_object_fullname(o):
        module = o.__class__.__module__
        if module is None or module == str.__class__.__module__:
            return o.__class__.__name__
        else:
            return module + '.' + o.__class__.__name__
