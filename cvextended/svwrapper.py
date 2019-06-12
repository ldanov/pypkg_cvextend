#!/usr/bin/env python3

"""A wrapper for methods from smote_variants to be useable within imblearn.pipeline.Pipeline"""

# Authors: Lyubomir Danov <->
# License: -

from sklearn.base import BaseEstimator
from collections import defaultdict


class SamplingWrapperSV(BaseEstimator):
    """
        Wrapper for methods from smote_variants package 
        to be useable as steps in imblearn.pipeline.Pipeline

        Maps the sv_sampler.sample(X, y) to self.fit_resample(X, y)
        Exposes a set_params() method

    """

    _valid_modules = ['smote_variants._smote_variants']
    _valid_methods = ['sample', 'get_params']

    def __init__(self, sv_sampler):
        self._validate_module(sv_sampler)
        self._validate_methods(sv_sampler)

        self._sv_sampler = sv_sampler

    def fit(self, X, y=None, sample_weight=None):
        return self

    def fit_resample(self, X, y):
        output = self._sv_sampler.sample(X, y)
        return output

    fit_sample = fit_resample

    def get_params(self, deep=True):
        out = self._sv_sampler.get_params()
        return out

    def set_params(self, **params):
        """Set the parameters of this estimator.

        The method works on simple estimators as well as on nested objects
        (such as pipelines). The latter have parameters of the form
        ``<component>__<parameter>`` so that it's possible to update each
        component of a nested object.

        Returns
        -------
        self
        """
        if not params:
            # Simple optimization to gain speed (inspect is slow)
            return self
        valid_params = self.get_params(deep=True)

        nested_params = defaultdict(dict)  # grouped by prefix
        for key, value in params.items():
            key, delim, sub_key = key.partition('__')
            if key not in valid_params:
                raise ValueError('Invalid parameter %s for estimator %s. '
                                 'Check the list of available parameters '
                                 'with `estimator.get_params().keys()`.' %
                                 (key, self))

            if delim:
                nested_params[key][sub_key] = value
            else:
                setattr(self._sv_sampler, key, value)
                valid_params[key] = value

        for key, sub_params in nested_params.items():
            valid_params[key].set_params(**sub_params)

        return self

    @classmethod
    def _validate_module(cls, to_check):
        allowed_types = cls._valid_modules
        found_module = to_check.__class__.__module__
        if not (found_module in allowed_types):
            raise TypeError('Class {a} only accepts object from {b}, not {c}!'
                             .format(a=cls.__class__, b=allowed_types, c=found_module))
        return True

    @classmethod
    def _validate_methods(cls, to_check):
        expected_methods = cls._valid_methods
        for method in expected_methods:
            method_impl = getattr(to_check, method, None)
            if method_impl is None:
                raise AttributeError('Object {a} does not have attribute {b}!'
                                     .format(a=cls.__class__, b=method))
            if not callable(method_impl):
                raise ValueError('Method {a} of object {b} is not callable!'
                                 .format(a=method, b=cls.__class__))
        return True
