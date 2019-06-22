#!/usr/bin/env python3

"""ScoreGrid is a utility class holding information about which score to use """

# Authors: Lyubomir Danov <->
# License: -

from sklearn.metrics import scorer


class ScoreGrid(object):
    _expected_keys = [
        {
            # user-defined name that will be used in generating result df columnname
            'name': 'score_name',
            'type': str
        },
        {
            # which column or key to use when looking for metric
            'name': 'score_key',
            'type': str
        },
        {
            # which pandas-known string callable to give to call transform on results
            'name':  'score_criteria',
            'type': str
        },
        {
            # scorer object itself
            'name': 'scorer',
            'type': scorer._BaseScorer
        }]

    def __init__(self, score_selection):
        for score in score_selection:
            for exp_k in self._expected_keys:
                if not isinstance(score[exp_k['name']], exp_k['type']):
                    raise TypeError()
        self.score_selection = score_selection

    def get_sklearn_dict(self):
        '''
        Returns a dict of scores as expected by sklearn.BaseSearchCV scoring param
        '''
        sklearn_score_dict = {}
        for score in self.score_selection:
            sklearn_score_dict[score['score_name']] = score['scorer']

        return sklearn_score_dict
