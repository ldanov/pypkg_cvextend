#!/usr/bin/env python3

"""ScoreGrid is a utility class holding information about which score to use """

# Authors: Lyubomir Danov <->
# License: -

from sklearn.metrics import scorer


class ScoreGrid(object):
    _expected_keys = [
        {
            'name': 'score_name',
            'type': str
        },
        {
            'name': 'score_criterion_name',
            'type': str
        },
        {
            'name':  'score_criterion_selector',
            'type': str
        },
        {
            'name': 'scorer',
            'type': scorer._BaseScorer
        },
        {
            'name': 'use_for_selection',
            'type': bool
        }]

    def __init__(self, score_selection):
        for score in score_selection:
            for exp_k in self._expected_keys:
                if not isinstance(score[exp_k['name']], exp_k['type']):
                    raise TypeError()
        self.score_selection = score_selection

    def get_sklearn_dict(self):
        '''
        Returns a dict of scores as expected by sklearn.GridSearchCV scoring param
        '''
        sklearn_score_dict = {}
        for score in self.score_selection:
            sklearn_score_dict[score['score_name']] = score['scorer']

        return sklearn_score_dict

    def get_selection_scores(self):
        '''
        returns array of dicts as used by NestedGrid
        '''
        selection_scores = []
        for score in self.score_selection:
            if score['use_for_selection']:
                selection_scores.append(score)
        return selection_scores
