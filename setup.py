#!/usr/bin/env python3

from setuptools import setup
from cvextend._version import __version__

__PROJECT_NAME__ = 'cvextend'
__DESCRIPTION__ = "Tools to extend sklearn's cross-validation classes and functions"
__AUTHORS__ = 'Lyubomir Danov'
__URL__ = 'https://github.com/ldanov/pypkg_cvextend'
__PACKAGES__ = [__PROJECT_NAME__]

setup(name=__PROJECT_NAME__,
      version=__version__,
      description=__DESCRIPTION__,
      author=__AUTHORS__,
      url=__URL__,
      packages=__PACKAGES__,
      install_requires = ['numpy == 1.16.3', 'scikit_learn == 0.21.2', 'pandas == 0.24.2']
     )