# Readme

## Description 

`cvextend` extends the functionality offered by the Model Selection module of [scikit-learn](https://scikit-learn.org/stable/), specifically aimed at better API for nested cross-validation.

`cvextend.generate_param_grid` helps with permutating models in steps and their hyperparameters in your `sklearn.pipeline`. See the example [here](https://pypkg-cvextend.readthedocs.io/en/latest/api/cvextend.generate_param_grid.html)

`cvextend.ScoreGrid` helps you keep track of the scores you want to use in cross-validations with different optimisation objectives in parallel. See the example [here](https://pypkg-cvextend.readthedocs.io/en/latest/api/cvextend.ScoreGrid.html) and also [cvextend.EvaluationGrid](https://pypkg-cvextend.readthedocs.io/en/latest/api/cvextend.ScoreGrid.html).

`cvextend.EvaluationGrid` supports searching multiple scores metrics and refits estimators to be passed to external CV loop.

`cvextend.nested_cv` enables performing nested cross-validation using scikit-learn Pipeline and CV instances, but allowing for multiple score optimisation in inner CV loop.

`cvextend.basic_cv` is a wrapper around a scikit-learn CV instance with requirement for Pipeline as estimator.

## Installation 

To install the cvextend library from [PyPI](https://pypi.org/project/cvextend/) use pip:

```
pip install cvextend
```

or install directly from source:

```
python setup.py install
```

## Usage

``` 
>>> from cvextend import nested_cv
>>> from cvextend import generate_param_grid
>>> from cvextend import ScoreGrid
>>> import pandas
>>> from sklearn.datasets import load_breast_cancer
>>> from sklearn.svm import SVC
>>> from sklearn.ensemble import RandomForestClassifier
>>> from sklearn.model_selection import GridSearchCV, StratifiedKFold
>>> from sklearn.pipeline import Pipeline
>>> steps = {
...     'preprocessor': {'skip': None},
...     'classifier': {
...         'svm': SVC(probability=True),
...         'rf': RandomForestClassifier()
...     }
... }
>>> param_dict = {
...     'skip': {},
...     'svm': {'C': [1, 10, 100],
...             'gamma': [.01, .1],
...             'kernel': ['rbf']},
...     'rf': {'n_estimators': [1, 10, 100],
...         'max_features': [1, 5, 10, 20]}
... }
>>> scorer_selection = ScoreGrid(scorers)
>>> sk_score = scorer_selection.get_sklearn_dict()
>>> pipe = Pipeline([('preprocessor', None), ('classifier', None)])
>>> X, y = load_breast_cancer(return_X_y=True)
>>> params, steps = generate_param_grid(steps=steps,
...                                     param_dict=param_dict)
>>> inner_cv_use = StratifiedKFold(n_splits=5, shuffle=True,
...                                random_state=0)
>>> inner_cv_seeds = [1,2]
>>> test_cv_grid = GridSearchCV(estimator=pipe,
...                             param_grid=params,
...                             scoring=sk_score,
...                             cv=inner_cv_use,
...                             refit=False)
>>> outer_cv_use = StratifiedKFold(n_splits=2, random_state=1,
...                                shuffle=True)
>>> addit_info = {'dataset_name': "breast_cancer"}
>>> result_outer, result_inner = nested_cv(cv_grid=test_cv_grid,
...                                        X=X, y=y,
...                                        score_selection=scorer_selection,
...                                        inner_cv_seeds=inner_cv_seeds,
...                                        outer_cv=outer_cv_use,
...                                        additional_info=addit_info
...                                        )
>>> print(pandas.DataFrame(result_outer))
>>> print(pandas.concat([pandas.DataFrame(x) for x in result_inner]))
```

## Questions and comments
In case of questions or comments, write an email:  
`ldanov@users.noreply.github.com`